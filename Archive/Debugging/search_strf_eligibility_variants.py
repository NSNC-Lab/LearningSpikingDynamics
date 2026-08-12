from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_difference_strf_eligibility import (  # noqa: E402
    finite_metrics,
    load_gt_data_for_test,
    make_test_args,
    perturb_params,
    select_params_from_config,
    set_seed,
)
from Pre_Processing import preprocess_handler  # noqa: E402
from Simulation import Architecture_Declaration, conditional_handler, ode_handler  # noqa: E402


PARAM_TO_RATE_DERIV = {
    "STRF_gain": ("onset_rate_gain_deriv", "offset_rate_gain_deriv"),
    "STRF_alpha": ("onset_rate_deriv", "offset_rate_deriv"),
}

PARAM_SHORT = {
    "STRF_gain": "gain",
    "STRF_alpha": "alpha",
}


def build_states(args: dict[str, Any], params: dict[str, np.ndarray]) -> dict[str, Any]:
    return Architecture_Declaration.build_network(args, torch.device(args["simulation"]["device"]), params)


def current_psth(states: dict[str, Any], args: dict[str, Any], timestep: int) -> np.ndarray:
    granularity = int(args["simulation"]["PSTH_granularity"])
    holder = states["neurons"]["Dynamic"]["ron"]["spikes_holder"][
        :, :, :, timestep - granularity : timestep
    ]
    return torch.sum(torch.sum(holder, dim=1), dim=-1).detach().cpu().numpy().astype(np.float64)


def run_forward_bins(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = build_states(args, params)
    with torch.no_grad():
        pre_processed = preprocess_handler.preprocess(args, states, device)
        bins = []
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(args, states, pre_processed["spks"], timestep)
            states = conditional_handler.run_conditionals(args, states, timestep)
            if timestep % int(args["simulation"]["PSTH_granularity"]) == 0 and timestep > 0:
                bins.append(current_psth(states, args, timestep)[0, 0])
    return np.asarray(bins, dtype=np.float64)


def finite_difference_target(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    seed: int,
    params_to_test: list[str],
) -> dict[str, Any]:
    base_values = {
        "STRF_gain": float(params["Strf_gain"][0, 0]),
        "STRF_alpha": float(params["Strf_alpha"][0, 0]),
    }
    epsilons = {
        "STRF_gain": max(abs(base_values["STRF_gain"]) * 0.05, 5e-5),
        "STRF_alpha": max(abs(base_values["STRF_alpha"]) * 0.05, 0.25),
    }

    target: dict[str, Any] = {}
    for param_name in params_to_test:
        epsilon = epsilons[param_name]
        plus = run_forward_bins(args, perturb_params(params, param_name, epsilon), seed)
        minus = run_forward_bins(args, perturb_params(params, param_name, -epsilon), seed)
        target[param_name] = {
            "epsilon": float(epsilon),
            "psth_fd": (plus - minus) / (2.0 * epsilon),
        }
    return target


def zeros_like_shape(args: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.zeros(
        (
            int(args["simulation"]["batch_size"]),
            10,
            len(args["simulation"]["cell_targets"]),
        ),
        dtype=torch.float32,
        device=device,
    )


def init_trace_state(args: dict[str, Any], states: dict[str, Any], device: torch.device) -> dict[str, Any]:
    trace: dict[str, Any] = {}
    for param_name in PARAM_SHORT.values():
        trace[param_name] = {"dV": {}, "dPSC_s": {}, "dPSC_x": {}, "buffers": {}, "bin": None}
        for neuron_name in states["neurons"]["Static"]:
            trace[param_name]["dV"][neuron_name] = zeros_like_shape(args, device)
        for synapse_name, syn_static in states["synapses"]["Static"].items():
            trace[param_name]["dPSC_s"][synapse_name] = zeros_like_shape(args, device)
            trace[param_name]["dPSC_x"][synapse_name] = zeros_like_shape(args, device)
            delay_steps = int(round(float(syn_static["PSC_delay"]) / float(args["simulation"]["dt"])))
            if delay_steps > 0:
                trace[param_name]["buffers"][synapse_name] = torch.zeros(
                    (delay_steps, *zeros_like_shape(args, device).shape),
                    dtype=torch.float32,
                    device=device,
                )
        trace[param_name]["bin"] = torch.zeros(
            (int(args["simulation"]["batch_size"]), len(args["simulation"]["cell_targets"])),
            dtype=torch.float32,
            device=device,
        )
    return trace


def rel_ref_probability(args: dict[str, Any], states: dict[str, Any], timestep: int) -> torch.Tensor:
    ron_dynamic = states["neurons"]["Dynamic"]["ron"]
    learnable = states["neurons"]["Learnable"]["ron"]
    last_spike = ron_dynamic["tspike"].max(dim=-1).values
    return torch.clamp(
        learnable["rel_ref_c"][:, None, :]
        * torch.tanh(learnable["rel_ref_a"][:, None, :] * (timestep - last_spike) - learnable["rel_ref_b"][:, None, :])
        + learnable["rel_ref_c"][:, None, :],
        min=0.0,
        max=1.0,
    )


def neuron_psi(states: dict[str, Any], neuron_name: str) -> torch.Tensor:
    static = states["neurons"]["Static"][neuron_name]
    dynamic = states["neurons"]["Dynamic"][neuron_name]
    return 1.0 - torch.tanh(dynamic["V"][:, :, :, -1] - static["V_thresh"]) ** 2


def delayed_presynaptic_derivative(
    trace_param: dict[str, Any],
    synapse_name: str,
    current_derivative: torch.Tensor,
) -> torch.Tensor:
    if synapse_name not in trace_param["buffers"]:
        return current_derivative
    buffer = trace_param["buffers"][synapse_name]
    delayed = buffer[0].clone()
    buffer[:-1] = buffer[1:].clone()
    buffer[-1] = current_derivative
    return delayed


def update_continuous_traces(
    args: dict[str, Any],
    states: dict[str, Any],
    pre_processed: dict[str, Any],
    trace: dict[str, Any],
    timestep: int,
    variant: dict[str, Any],
) -> dict[str, torch.Tensor]:
    dt = float(args["simulation"]["dt"])
    device = torch.device(args["simulation"]["device"])
    rate_object = pre_processed["onset_offset_rates"]
    input_spikes = pre_processed["spks"]["onset_offset_spks"]
    spike_derivatives: dict[str, torch.Tensor] = {}

    for param_name, short_name in PARAM_SHORT.items():
        onset_deriv_key, offset_deriv_key = PARAM_TO_RATE_DERIV[param_name]
        input_rate_derivs = {
            "on": rate_object[onset_deriv_key][timestep, :, None, :].to(device),
            "off": rate_object[offset_deriv_key][timestep, :, None, :].to(device),
        }
        trace_param = trace[short_name]
        old_dv = {name: value.clone() for name, value in trace_param["dV"].items()}
        old_dpsc_s = {name: value.clone() for name, value in trace_param["dPSC_s"].items()}
        old_dpsc_x = {name: value.clone() for name, value in trace_param["dPSC_x"].items()}

        for neuron_name in states["neurons"]["Static"]:
            static = states["neurons"]["Static"][neuron_name]
            dynamic = states["neurons"]["Dynamic"][neuron_name]
            v_old = dynamic["V"][:, :, :, -2]
            g_ad_old = dynamic["g_ad"][:, :, :, -2]
            d_v = old_dv[neuron_name]

            if static["input"] == 1:
                spike_key = f"{neuron_name}set_spks"
                input_spike = input_spikes[spike_key][:, :, :, timestep].to(torch.float32)
                d_input_spike = (dt / 1000.0) * input_rate_derivs[neuron_name]
                rhs = (
                    (-(1.0 + static["R"] * g_ad_old) / static["tau"])
                    - (static["R"] * static["g_postIC"] * input_spike / static["tau"])
                ) * d_v
                rhs = rhs - (
                    static["R"]
                    * static["g_postIC"]
                    * (v_old - static["E_exc"])
                    * d_input_spike
                    / static["tau"]
                )
                trace_param["dV"][neuron_name] = d_v + dt * rhs
            else:
                rhs = (-(1.0 + static["R"] * g_ad_old) / static["tau"]) * d_v
                for synapse_name in static["projections"]:
                    syn_static = states["synapses"]["Static"][synapse_name]
                    syn_learnable = states["synapses"]["Learnable"][synapse_name]
                    psc_old = states["synapses"]["Dynamic"][synapse_name]["PSC_s"][:, :, :, -2]
                    rhs = rhs - (
                        static["R"]
                        * syn_learnable["gSYN"][:, None, :]
                        * (
                            psc_old * d_v
                            + old_dpsc_s[synapse_name] * (v_old - syn_static["ESYN"])
                        )
                        / static["tau"]
                    )
                if static["noise"] == 1:
                    noise_sn = dynamic["noise_sn"][:, :, :, -2]
                    rhs = rhs - static["R"] * static["nSYN"] * noise_sn * d_v / static["tau"]
                trace_param["dV"][neuron_name] = d_v + dt * rhs

        for synapse_name, syn_static in states["synapses"]["Static"].items():
            trace_param["dPSC_s"][synapse_name] = old_dpsc_s[synapse_name] + dt * (
                syn_static["scale"] * old_dpsc_x[synapse_name] - old_dpsc_s[synapse_name]
            ) / syn_static["tauR"]
            trace_param["dPSC_x"][synapse_name] = old_dpsc_x[synapse_name] + dt * (
                -old_dpsc_x[synapse_name] / syn_static["tauD"]
            )

        for neuron_name in states["neurons"]["Static"]:
            psi = neuron_psi(states, neuron_name)
            scale = float(variant["psi_scale"])
            if neuron_name == "ron" and variant["output_relprob"]:
                scale_tensor = scale * rel_ref_probability(args, states, timestep)
            else:
                scale_tensor = scale
            spike_derivatives[f"{short_name}_{neuron_name}"] = scale_tensor * psi * trace_param["dV"][neuron_name]

        if variant["output_source"] == "spike":
            trace_param["bin"] = trace_param["bin"] + torch.sum(
                spike_derivatives[f"{short_name}_ron"], dim=1
            )
        elif variant["output_source"] == "voltage":
            trace_param["bin"] = trace_param["bin"] + torch.sum(
                trace_param["dV"]["ron"], dim=1
            )
        else:
            raise ValueError(f"Unknown output_source: {variant['output_source']}")

    return spike_derivatives


def apply_condition_derivative_updates(
    args: dict[str, Any],
    states: dict[str, Any],
    trace: dict[str, Any],
    spike_derivatives: dict[str, torch.Tensor],
    q_before: dict[str, torch.Tensor],
    variant: dict[str, Any],
) -> None:
    for short_name in PARAM_SHORT.values():
        trace_param = trace[short_name]
        if variant["reset_dv"]:
            for neuron_name, static in states["neurons"]["Static"].items():
                v_now = states["neurons"]["Dynamic"][neuron_name]["V"][:, :, :, -1]
                reset_mask = v_now == static["V_reset"]
                trace_param["dV"][neuron_name] = torch.where(
                    reset_mask, torch.zeros_like(trace_param["dV"][neuron_name]), trace_param["dV"][neuron_name]
                )

        for synapse_name in states["synapses"]["Static"]:
            pre_name = synapse_name.split("_")[0]
            delayed = delayed_presynaptic_derivative(
                trace_param,
                synapse_name,
                spike_derivatives[f"{short_name}_{pre_name}"],
            )
            trace_param["dPSC_x"][synapse_name] = (
                trace_param["dPSC_x"][synapse_name] + q_before[synapse_name] * delayed
            )


def run_surrogate_bins(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    seed: int,
    variant: dict[str, Any],
) -> dict[str, np.ndarray]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = build_states(args, params)
    trace = init_trace_state(args, states, device)
    outputs = {"STRF_gain": [], "STRF_alpha": []}

    with torch.no_grad():
        pre_processed = preprocess_handler.preprocess(args, states, device)
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(args, states, pre_processed["spks"], timestep)
            for neuron_name in states["neurons"]["Static"]:
                states["neurons"]["Static"][neuron_name]["psi"] = neuron_psi(states, neuron_name)

            q_before = {
                synapse_name: states["synapses"]["Dynamic"][synapse_name]["PSC_q"][:, :, :, -1].clone()
                for synapse_name in states["synapses"]["Static"]
            }
            spike_derivatives = update_continuous_traces(
                args, states, pre_processed, trace, timestep, variant
            )
            states = conditional_handler.run_conditionals(args, states, timestep)
            apply_condition_derivative_updates(
                args, states, trace, spike_derivatives, q_before, variant
            )

            if timestep % int(args["simulation"]["PSTH_granularity"]) == 0 and timestep > 0:
                for param_name, short_name in PARAM_SHORT.items():
                    outputs[param_name].append(float(trace[short_name]["bin"][0, 0].detach().cpu()))
                    trace[short_name]["bin"].zero_()

    return {key: np.asarray(value, dtype=np.float64) for key, value in outputs.items()}


def make_variants() -> list[dict[str, Any]]:
    variants = []
    for psi_scale in (1.0, 0.5):
        for output_relprob in (False, True):
            for reset_dv in (False, True):
                variants.append(
                    {
                        "name": (
                            f"full_dv_psi{psi_scale:g}_"
                            f"{'relprob' if output_relprob else 'norel'}_"
                            f"{'reset' if reset_dv else 'noreset'}"
                        ),
                        "psi_scale": psi_scale,
                        "output_relprob": output_relprob,
                        "reset_dv": reset_dv,
                        "output_source": "spike",
                    }
                )
    variants.append(
        {
            "name": "full_dv_voltage_output",
            "psi_scale": 1.0,
            "output_relprob": False,
            "reset_dv": True,
            "output_source": "voltage",
        }
    )
    return variants


def compare_variant(
    variant: dict[str, Any],
    surrogate: dict[str, np.ndarray],
    target: dict[str, Any],
    baseline_bins: np.ndarray,
    data_bins: np.ndarray,
) -> dict[str, Any]:
    result = {"name": variant["name"], "settings": variant, "params": {}}
    gradient = 2.0 * (baseline_bins - data_bins)
    for param_name, values in surrogate.items():
        psth_fd = target[param_name]["psth_fd"]
        loss_fd = float(np.dot(gradient, psth_fd))
        surrogate_grad = float(np.dot(gradient, values))
        metrics = finite_metrics(values, psth_fd)
        result["params"][param_name] = {
            "loss_fd_grad_from_bins": loss_fd,
            "surrogate_grad_from_bins": surrogate_grad,
            "same_sign": bool(np.sign(loss_fd) == np.sign(surrogate_grad)),
            "metrics": metrics,
        }
    result["score"] = float(
        np.nanmean(
            [
                result["params"][param]["metrics"]["relative_rmse"]
                for param in result["params"]
            ]
        )
    )
    result["mean_correlation"] = float(
        np.nanmean(
            [
                result["params"][param]["metrics"]["correlation"]
                for param in result["params"]
            ]
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive-only STRF eligibility variant search.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=3001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "strf_eligibility_variant_search.json",
    )
    args_cli = parser.parse_args()

    with args_cli.config.open("r", encoding="utf-8") as handle:
        base_args = yaml.safe_load(handle)

    if args_cli.device.startswith("cuda") and not torch.cuda.is_available():
        args_cli.device = "cpu"

    test_args = make_test_args(
        base_args,
        cell_target=args_cli.cell,
        batch_size=args_cli.batch_size,
        sim_len=args_cli.sim_len,
        device=args_cli.device,
    )
    params = select_params_from_config(base_args, test_args, args_cli.cell, args_cli.batch)
    gt_data = load_gt_data_for_test(test_args)
    bins = int(args_cli.sim_len // int(test_args["simulation"]["PSTH_granularity"]))
    data_bins = gt_data["psth_holder"][0, :bins].detach().cpu().numpy().astype(np.float64)

    print("Building finite-difference target...")
    baseline_bins = run_forward_bins(test_args, params, args_cli.seed)
    target = finite_difference_target(test_args, params, args_cli.seed, ["STRF_gain", "STRF_alpha"])

    variants = make_variants()
    if args_cli.max_variants > 0:
        variants = variants[: args_cli.max_variants]

    results = []
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] Testing {variant['name']}")
        surrogate = run_surrogate_bins(test_args, params, args_cli.seed, variant)
        results.append(compare_variant(variant, surrogate, target, baseline_bins, data_bins))

    results.sort(key=lambda item: (item["score"], -item["mean_correlation"]))
    output = {
        "cell": args_cli.cell,
        "batch": args_cli.batch,
        "sim_len": args_cli.sim_len,
        "seed": args_cli.seed,
        "baseline_bins": baseline_bins.tolist(),
        "data_bins": data_bins.tolist(),
        "target": {
            param: {
                "epsilon": target[param]["epsilon"],
                "psth_fd": target[param]["psth_fd"].tolist(),
            }
            for param in target
        },
        "results": results,
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(results[:3], indent=2))
    print(f"Saved results to {args_cli.output}")


if __name__ == "__main__":
    main()
