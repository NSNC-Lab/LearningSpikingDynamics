from __future__ import annotations

import argparse
import copy
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
from Simulation import (  # noqa: E402
    Architecture_Declaration,
    Eligibility_handler,
    Loss_handler,
    conditional_handler,
    ode_handler,
)
from Pre_Processing import preprocess_handler  # noqa: E402


PARAM_MAP = {
    "STRF_gain": ("Strf_gain", "STRF_gain_grad"),
    "STRF_alpha": ("Strf_alpha", "STRF_alpha_grad"),
}


def clone_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in params.items()}


def perturb(params: dict[str, np.ndarray], param: str, delta: float) -> dict[str, np.ndarray]:
    out = clone_params(params)
    init_key, _ = PARAM_MAP[param]
    out[init_key][0, 0] = np.float32(out[init_key][0, 0] + delta)
    if init_key == "Strf_gain":
        out[init_key][0, 0] = max(out[init_key][0, 0], np.float32(0.001))
    if init_key == "Strf_alpha":
        out[init_key][0, 0] = np.clip(out[init_key][0, 0], np.float32(0.1), np.float32(250.0))
    return out


def run_epoch(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    gt_data: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = Architecture_Declaration.build_network(args, device, params)

    with torch.no_grad():
        pre_processed = preprocess_handler.preprocess(args, states, device)
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(args, states, pre_processed["spks"], timestep)
            states = Eligibility_handler.update_eligibility(
                args, states, pre_processed["onset_offset_rates"], timestep
            )
            states = conditional_handler.run_conditionals(args, states, timestep)
            states = Loss_handler.handle_loss(args, states, gt_data, timestep, epoch=0)

    spikes = states["neurons"]["Dynamic"]["ron"]["spikes_holder"].to(torch.float32)
    duration_s = int(args["simulation"]["sim_len"]) * float(args["simulation"]["dt"]) / 1000.0
    firing_rate = float(spikes.sum().detach().cpu()) / (
        int(args["simulation"]["batch_size"]) * 10 * len(args["simulation"]["cell_targets"]) * duration_s
    )
    return {
        "loss": float(states["neurons"]["Dynamic"]["ron"]["mean_sse_loss"]),
        "firing_rate_hz": firing_rate,
        "STRF_gain_grad": float(states["neurons"]["Learnable"]["STRF_gain_grad"][0, 0].detach().cpu()),
        "STRF_alpha_grad": float(states["neurons"]["Learnable"]["STRF_alpha_grad"][0, 0].detach().cpu()),
    }


def check_param(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    baseline: dict[str, Any],
    gt_data: dict[str, torch.Tensor],
    param: str,
    seed: int,
    step: float,
) -> dict[str, Any]:
    _, grad_key = PARAM_MAP[param]
    grad = baseline[grad_key]
    plus = run_epoch(args, perturb(params, param, step), gt_data, seed)
    minus = run_epoch(args, perturb(params, param, -step), gt_data, seed)
    along_update = minus if grad > 0 else plus
    against_update = plus if grad > 0 else minus
    fd_grad = (plus["loss"] - minus["loss"]) / (2.0 * step)
    return {
        "base_value": float(params[PARAM_MAP[param][0]][0, 0]),
        "test_step": step,
        "analytic_grad": grad,
        "finite_difference_loss_grad": float(fd_grad),
        "same_sign_analytic_vs_fd": bool(np.sign(grad) == np.sign(fd_grad)),
        "loss_plus": plus["loss"],
        "loss_minus": minus["loss"],
        "loss_after_negative_analytic_step": along_update["loss"],
        "loss_after_positive_analytic_step": against_update["loss"],
        "firing_rate_after_negative_analytic_step_hz": along_update["firing_rate_hz"],
        "firing_rate_after_positive_analytic_step_hz": against_update["firing_rate_hz"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check STRF gradient update direction against PSTH SSE.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gain-step", type=float, default=1e-4)
    parser.add_argument("--alpha-step", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "strf_update_direction_check.json",
    )
    cli = parser.parse_args()

    with cli.config.open("r", encoding="utf-8") as handle:
        base_args = yaml.safe_load(handle)
    args = make_test_args(base_args, cli.cell, 1, cli.sim_len, cli.device)
    args.setdefault("debug_plots", {})["enabled"] = False
    params = select_params_from_config(base_args, args, cli.cell, cli.batch)
    gt_data = load_gt_data_for_test(args)

    baseline = run_epoch(args, params, gt_data, cli.seed)
    result = {
        "cell": cli.cell,
        "batch": cli.batch,
        "sim_len": cli.sim_len,
        "seed": cli.seed,
        "baseline": baseline,
        "checks": {
            "STRF_gain": check_param(
                args, params, baseline, gt_data, "STRF_gain", cli.seed, cli.gain_step
            ),
            "STRF_alpha": check_param(
                args, params, baseline, gt_data, "STRF_alpha", cli.seed, cli.alpha_step
            ),
        },
    }

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved results to {cli.output}")


if __name__ == "__main__":
    main()
