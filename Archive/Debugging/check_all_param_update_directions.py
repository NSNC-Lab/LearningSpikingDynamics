from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Archive.Debugging.finite_difference_strf_eligibility import (  # noqa: E402
    load_gt_data_for_test,
    make_test_args,
    select_params_from_config,
    set_seed,
)
from Pre_Processing import preprocess_handler  # noqa: E402
from Simulation import (  # noqa: E402
    Architecture_Declaration,
    Eligibility_handler,
    Loss_handler,
    conditional_handler,
    ode_handler,
)


PARAMS = [
    {"name": "STRF_gain", "init": "Strf_gain", "loss": "psth", "step": 1e-4, "min": 0.001},
    {"name": "STRF_alpha", "init": "Strf_alpha", "loss": "psth", "step": 0.25, "min": 0.1, "max": 250.0},
    {"name": "output_ad", "init": "output_ad", "loss": "psth", "step": 1e-5, "min": 0.0},
    {"name": "on_ron_gSYN", "init": "on_ron_gSYN", "loss": "psth", "step": 1e-4, "min": 0.001},
    {"name": "off_ron_gSYN", "init": "off_ron_gSYN", "loss": "psth", "step": 1e-4, "min": 0.001},
    {"name": "on_sonoff_gSYN", "init": "on_sonoff_gSYN", "loss": "psth", "step": 1e-4, "min": 0.001},
    {"name": "off_sonoff_gSYN", "init": "off_sonoff_gSYN", "loss": "psth", "step": 1e-4, "min": 0.001},
    {"name": "sonoff_ron_gSYN", "init": "sonoff_ron_gSYN", "loss": "psth", "step": 1e-4, "min": 0.001},
    {"name": "abs_ref", "init": "abs_ref", "loss": "cv", "step": 0.05, "min": 0.0},
    {"name": "rel_ref_a", "init": "rel_ref_a", "loss": "cv", "step": 0.05, "min": 0.001},
    {"name": "rel_ref_b", "init": "rel_ref_b", "loss": "cv", "step": 0.05, "min": 0.0},
    {"name": "rel_ref_c", "init": "rel_ref_c", "loss": "cv", "step": 0.01, "min": 0.2, "max": 1.0},
]

OPTIMIZER_SCALE_STEPS = {
    "STRF_gain": 0.001,
    "STRF_alpha": 5.0,
    "output_ad": 0.00025,
    "on_ron_gSYN": 0.0025,
    "off_ron_gSYN": 0.0025,
    "on_sonoff_gSYN": 0.0025,
    "off_sonoff_gSYN": 0.0025,
    "sonoff_ron_gSYN": 0.0025,
    "abs_ref": 2.0,
    "rel_ref_a": 1.0,
    "rel_ref_b": 2.0,
    "rel_ref_c": 0.02,
}


def clone_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in params.items()}


def get_param_value(params: dict[str, np.ndarray], spec: dict[str, Any]) -> float:
    return float(params[spec["init"]][0, 0])


def choose_step(params: dict[str, np.ndarray], spec: dict[str, Any]) -> tuple[float, str]:
    value = get_param_value(params, spec)
    nominal_step = max(float(spec["step"]), abs(value) * 0.01)
    lower = spec.get("min", -np.inf)
    upper = spec.get("max", np.inf)
    lower_room = value - lower if np.isfinite(lower) else np.inf
    upper_room = upper - value if np.isfinite(upper) else np.inf
    central_room = min(lower_room, upper_room)

    if central_room > 0:
        return float(min(nominal_step, central_room * 0.5)), "central"
    if upper_room > 0:
        return float(min(nominal_step, upper_room * 0.5)), "forward"
    if lower_room > 0:
        return float(min(nominal_step, lower_room * 0.5)), "backward"
    return float(nominal_step), "unconstrained_fallback"


def perturb(params: dict[str, np.ndarray], spec: dict[str, Any], delta: float) -> dict[str, np.ndarray]:
    out = clone_params(params)
    key = spec["init"]
    out[key][0, 0] = np.float32(out[key][0, 0] + delta)
    return out


def as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def get_grad(states: dict[str, Any], name: str) -> float:
    if name == "STRF_gain":
        return as_float(states["neurons"]["Learnable"]["STRF_gain_grad"][0, 0])
    if name == "STRF_alpha":
        return as_float(states["neurons"]["Learnable"]["STRF_alpha_grad"][0, 0])
    if name == "output_ad":
        return as_float(states["neurons"]["Learnable"]["ron"]["output_ad_grad"][0, 0])
    if name.endswith("_gSYN"):
        syn_name = name.removesuffix("_gSYN")
        return as_float(states["synapses"]["Learnable"][syn_name]["gSYN_grad"][0, 0])
    return as_float(states["neurons"]["Learnable"]["ron"][f"{name}_grad"][0, 0])


def cv_loss_value(spikes_holder: torch.Tensor, gt_raster: torch.Tensor) -> float:
    losses = []
    batch_size = spikes_holder.shape[0]
    n_cells = spikes_holder.shape[2]
    for batch_index in range(batch_size):
        for cell_index in range(n_cells):
            sim_isis = []
            data_isis = []
            for trial in range(spikes_holder.shape[1]):
                sim_spikes = torch.where(spikes_holder[batch_index, trial, cell_index, :] == 1)[0]
                data_spikes = torch.where(gt_raster[cell_index, trial, :] == 1)[0]
                if len(sim_spikes) > 0:
                    sim_isis.append(torch.diff(sim_spikes))
                data_isis.append(torch.diff(data_spikes))
            if len(sim_isis) <= 1:
                continue
            sim_cat = torch.cat(sim_isis).to(torch.float32)
            data_cat = torch.cat(data_isis).to(torch.float32)
            if len(sim_cat) <= 1 or torch.mean(sim_cat) == 0:
                continue
            sim_cv = torch.std(sim_cat) / torch.mean(sim_cat)
            data_cv = torch.std(data_cat) / torch.mean(data_cat)
            losses.append((sim_cv - data_cv) ** 2)
    if not losses:
        return 0.0
    return as_float(torch.sum(torch.stack(losses)))


def run_epoch(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    gt_data: dict[str, torch.Tensor],
    seed: int,
    need_grad: bool,
) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = Architecture_Declaration.build_network(args, device, params)
    psth_loss = 0.0

    with torch.no_grad():
        pre_processed = preprocess_handler.preprocess(args, states, device)
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(args, states, pre_processed["spks"], timestep)
            if need_grad:
                states = Eligibility_handler.update_eligibility(
                    args, states, pre_processed["onset_offset_rates"], timestep
                )
            states = conditional_handler.run_conditionals(args, states, timestep)

            if need_grad:
                states = Loss_handler.handle_loss(args, states, gt_data, timestep, epoch=0)
            elif timestep % int(args["simulation"]["PSTH_granularity"]) == 0 and timestep > 0:
                granularity = int(args["simulation"]["PSTH_granularity"])
                holder = states["neurons"]["Dynamic"]["ron"]["spikes_holder"][
                    :, :, :, timestep - granularity : timestep
                ]
                sim_psth = torch.sum(torch.sum(holder, dim=1, keepdim=False), dim=-1, keepdim=False)
                data_psth = gt_data["psth_holder"][:, int(timestep / granularity) - 1]
                psth_loss += as_float(torch.mean((sim_psth.to(torch.float32) - data_psth[None, :]) ** 2))

    spikes = states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
    duration_s = int(args["simulation"]["sim_len"]) * float(args["simulation"]["dt"]) / 1000.0
    firing_rate = as_float(spikes.sum()) / (
        int(args["simulation"]["batch_size"]) * 10 * len(args["simulation"]["cell_targets"]) * duration_s
    )
    if need_grad:
        psth_loss = as_float(states["neurons"]["Dynamic"]["ron"]["mean_sse_loss"])
        cv_loss = as_float(states["neurons"]["Dynamic"]["ron"]["mean_CV_loss"])
        grads = {spec["name"]: get_grad(states, spec["name"]) for spec in PARAMS}
    else:
        cv_loss = cv_loss_value(spikes, gt_data["raster_holder"])
        grads = {}

    return {
        "psth_loss": psth_loss,
        "cv_loss": cv_loss,
        "firing_rate_hz": firing_rate,
        "grads": grads,
    }


def finite_difference_param(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    gt_data: dict[str, torch.Tensor],
    baseline: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    step, step_mode = choose_step(params, spec)
    objective = spec["loss"]
    loss_key = f"{objective}_loss"
    analytic_grad = float(baseline["grads"][spec["name"]])

    plus = None
    minus = None
    if step_mode in ("central", "forward", "unconstrained_fallback"):
        plus = run_epoch(args, perturb(params, spec, step), gt_data, seed, need_grad=False)
    if step_mode in ("central", "backward", "unconstrained_fallback"):
        minus = run_epoch(args, perturb(params, spec, -step), gt_data, seed, need_grad=False)

    if step_mode == "central":
        fd_grad = (plus[loss_key] - minus[loss_key]) / (2.0 * step)
    elif step_mode == "forward":
        fd_grad = (plus[loss_key] - baseline[loss_key]) / step
    elif step_mode == "backward":
        fd_grad = (baseline[loss_key] - minus[loss_key]) / step
    else:
        fd_grad = (plus[loss_key] - minus[loss_key]) / (2.0 * step)

    if analytic_grad > 0:
        negative_step = minus if minus is not None else baseline
        positive_step = plus if plus is not None else baseline
    else:
        negative_step = plus if plus is not None else baseline
        positive_step = minus if minus is not None else baseline

    return {
        "parameter": spec["name"],
        "objective": objective,
        "base_value": get_param_value(params, spec),
        "step": step,
        "step_mode": step_mode,
        "analytic_grad": analytic_grad,
        "finite_difference_grad": float(fd_grad),
        "same_sign": bool(np.sign(analytic_grad) == np.sign(fd_grad)),
        "baseline_objective_loss": baseline[loss_key],
        "loss_plus": plus[loss_key] if plus is not None else None,
        "loss_minus": minus[loss_key] if minus is not None else None,
        "loss_after_negative_analytic_step": negative_step[loss_key],
        "loss_after_positive_analytic_step": positive_step[loss_key],
        "negative_step_improved": bool(negative_step[loss_key] < baseline[loss_key]),
        "positive_step_worsened": bool(positive_step[loss_key] > baseline[loss_key]),
        "baseline_psth_loss": baseline["psth_loss"],
        "baseline_cv_loss": baseline["cv_loss"],
        "plus_psth_loss": plus["psth_loss"] if plus is not None else None,
        "minus_psth_loss": minus["psth_loss"] if minus is not None else None,
        "plus_cv_loss": plus["cv_loss"] if plus is not None else None,
        "minus_cv_loss": minus["cv_loss"] if minus is not None else None,
        "baseline_firing_rate_hz": baseline["firing_rate_hz"],
        "plus_firing_rate_hz": plus["firing_rate_hz"] if plus is not None else None,
        "minus_firing_rate_hz": minus["firing_rate_hz"] if minus is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-difference update-direction check for all 12 parameters.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--step-preset",
        choices=["local", "optimizer"],
        default="local",
        help="Use local finite-difference steps or approximate first-Adam-step scales.",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="Optional subset of parameter names to check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "all_param_update_direction_check.json",
    )
    args_cli = parser.parse_args()

    with args_cli.config.open("r", encoding="utf-8") as handle:
        base_args = yaml.safe_load(handle)
    args = make_test_args(base_args, args_cli.cell, 1, args_cli.sim_len, args_cli.device)
    args.setdefault("debug_plots", {})["enabled"] = False
    params = select_params_from_config(base_args, args, args_cli.cell, args_cli.batch)
    gt_data = load_gt_data_for_test(args)
    selected_specs = [dict(spec) for spec in PARAMS if args_cli.params is None or spec["name"] in args_cli.params]
    if args_cli.step_preset == "optimizer":
        for spec in selected_specs:
            spec["step"] = OPTIMIZER_SCALE_STEPS[spec["name"]]

    print("Running baseline with eligibility gradients...", flush=True)
    baseline = run_epoch(args, params, gt_data, args_cli.seed, need_grad=True)
    results = []
    for index, spec in enumerate(selected_specs, start=1):
        print(f"[{index}/{len(selected_specs)}] Checking {spec['name']} against {spec['loss']} loss...", flush=True)
        results.append(finite_difference_param(args, params, gt_data, baseline, spec, args_cli.seed))

    output = {
        "cell": args_cli.cell,
        "batch": args_cli.batch,
        "sim_len": args_cli.sim_len,
        "seed": args_cli.seed,
        "step_preset": args_cli.step_preset,
        "params_checked": [spec["name"] for spec in selected_specs],
        "baseline": {
            "psth_loss": baseline["psth_loss"],
            "cv_loss": baseline["cv_loss"],
            "firing_rate_hz": baseline["firing_rate_hz"],
        },
        "results": results,
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)
    print(f"Saved results to {args_cli.output}", flush=True)


if __name__ == "__main__":
    main()
