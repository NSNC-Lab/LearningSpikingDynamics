from __future__ import annotations

import argparse
import json
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

from check_direct_onoff_ron_path import (  # noqa: E402
    delayed,
    local_pre_spike_derivative,
    neuron_psi,
    zeros_trace,
)
from finite_difference_strf_eligibility import (  # noqa: E402
    finite_metrics,
    make_test_args,
    select_params_from_config,
    set_seed,
)
from Pre_Processing import preprocess_handler  # noqa: E402
from Simulation import Architecture_Declaration, Eligibility_handler, conditional_handler, ode_handler  # noqa: E402


PARAM_TO_DERIV = {
    "STRF_gain": {"on": "onset_rate_gain_deriv", "off": "offset_rate_gain_deriv"},
    "STRF_alpha": {"on": "onset_rate_deriv", "off": "offset_rate_deriv"},
}
PARAM_TO_SHORT = {"STRF_gain": "gain", "STRF_alpha": "alpha"}
INDIRECT_SYNAPSES = ("on_sonoff", "off_sonoff", "sonoff_ron")


def build_states(args: dict[str, Any], params: dict[str, np.ndarray]) -> dict[str, Any]:
    return Architecture_Declaration.build_network(args, torch.device(args["simulation"]["device"]), params)


def init_trace(args: dict[str, Any], device: torch.device) -> dict[str, Any]:
    trace: dict[str, Any] = {"buffers": {}}
    for param_name in PARAM_TO_DERIV:
        trace[param_name] = {}
        for synapse_name in INDIRECT_SYNAPSES:
            trace[param_name][synapse_name] = {
                "dPSC_s": zeros_trace(args, device),
                "dPSC_x": zeros_trace(args, device),
                "dV": zeros_trace(args, device),
            }
    return trace


def post_voltage_jacobian(
    args: dict[str, Any],
    states: dict[str, Any],
    post_name: str,
) -> torch.Tensor:
    post_static = states["neurons"]["Static"][post_name]
    post_dynamic = states["neurons"]["Dynamic"][post_name]
    jacobian = (-(1.0 + post_static["R"] * post_dynamic["g_ad"][:, :, :, -2]) / post_static["tau"])
    for projection_name in post_static["projections"]:
        projection_psc = states["synapses"]["Dynamic"][projection_name]["PSC_s"][:, :, :, -2]
        projection_gsyn = states["synapses"]["Learnable"][projection_name]["gSYN"][:, None, :]
        jacobian = jacobian - post_static["R"] * projection_psc * projection_gsyn / post_static["tau"]
    if post_static["noise"] == 1:
        noise_sn = post_dynamic["noise_sn"][:, :, :, -2]
        jacobian = jacobian - post_static["R"] * post_static["nSYN"] * noise_sn / post_static["tau"]
    return jacobian


def advance_trace_dynamics(
    args: dict[str, Any],
    states: dict[str, Any],
    trace: dict[str, Any],
    param_name: str,
    synapse_name: str,
    post_name: str,
) -> None:
    dt = float(args["simulation"]["dt"])
    syn_static = states["synapses"]["Static"][synapse_name]
    syn_learnable = states["synapses"]["Learnable"][synapse_name]
    post_static = states["neurons"]["Static"][post_name]
    post_dynamic = states["neurons"]["Dynamic"][post_name]
    cur_trace = trace[param_name][synapse_name]

    old_dpsc_s = cur_trace["dPSC_s"].clone()
    old_dpsc_x = cur_trace["dPSC_x"].clone()
    old_dv = cur_trace["dV"].clone()
    v_old = post_dynamic["V"][:, :, :, -2]

    d_v_rhs = post_voltage_jacobian(args, states, post_name) * old_dv
    d_v_rhs = d_v_rhs - (
        post_static["R"]
        * syn_learnable["gSYN"][:, None, :]
        * old_dpsc_s
        * (v_old - syn_static["ESYN"])
        / post_static["tau"]
    )

    cur_trace["dV"] = old_dv + dt * d_v_rhs
    cur_trace["dPSC_s"] = old_dpsc_s + dt * (
        syn_static["scale"] * old_dpsc_x - old_dpsc_s
    ) / syn_static["tauR"]
    cur_trace["dPSC_x"] = old_dpsc_x + dt * (-old_dpsc_x / syn_static["tauD"])


def inject_delayed_pre_derivative(
    args: dict[str, Any],
    states: dict[str, Any],
    trace: dict[str, Any],
    param_name: str,
    synapse_name: str,
    pre_deriv: torch.Tensor,
    q_before: torch.Tensor,
) -> None:
    dt = float(args["simulation"]["dt"])
    syn_static = states["synapses"]["Static"][synapse_name]
    delay_steps = int(round(float(syn_static["PSC_delay"]) / dt))
    delayed_pre = delayed(trace, f"{param_name}_{synapse_name}", pre_deriv, delay_steps)
    trace[param_name][synapse_name]["dPSC_x"] = (
        trace[param_name][synapse_name]["dPSC_x"] + q_before * delayed_pre
    )


def live_indirect_output(
    states: dict[str, Any],
    param_name: str,
) -> torch.Tensor:
    param_short = PARAM_TO_SHORT[param_name]
    return torch.sum(
        neuron_psi(states, "ron")
        * states["synapses"]["Dynamic"]["sonoff_ron"][f"dV_{param_short}"][:, :, :, -1],
        dim=1,
    )


def stateful_indirect_output(
    states: dict[str, Any],
    trace: dict[str, Any],
    param_name: str,
) -> torch.Tensor:
    return torch.sum(neuron_psi(states, "ron") * trace[param_name]["sonoff_ron"]["dV"], dim=1)


def collect(args: dict[str, Any], params: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = build_states(args, params)
    trace = init_trace(args, device)
    output = {
        param_name: {"stateful_indirect": [], "live_indirect": []}
        for param_name in PARAM_TO_DERIV
    }

    with torch.no_grad():
        pre_processed = preprocess_handler.preprocess(args, states, device)
        rate_object = pre_processed["onset_offset_rates"]
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(args, states, pre_processed["spks"], timestep)
            for neuron_name in states["neurons"]["Static"]:
                states["neurons"]["Static"][neuron_name]["psi"] = neuron_psi(states, neuron_name)

            q_before = {
                synapse_name: states["synapses"]["Dynamic"][synapse_name]["PSC_q"][:, :, :, -1].clone()
                for synapse_name in INDIRECT_SYNAPSES
            }

            for param_name in PARAM_TO_DERIV:
                advance_trace_dynamics(args, states, trace, param_name, "on_sonoff", "sonoff")
                advance_trace_dynamics(args, states, trace, param_name, "off_sonoff", "sonoff")
                advance_trace_dynamics(args, states, trace, param_name, "sonoff_ron", "ron")

                output[param_name]["stateful_indirect"].append(
                    stateful_indirect_output(states, trace, param_name).detach().cpu().numpy()
                )
                output[param_name]["live_indirect"].append(
                    live_indirect_output(states, param_name).detach().cpu().numpy()
                )

                on_pre_deriv = local_pre_spike_derivative(
                    args, states, rate_object, timestep, param_name, "on"
                )
                off_pre_deriv = local_pre_spike_derivative(
                    args, states, rate_object, timestep, param_name, "off"
                )
                sonoff_pre_deriv = neuron_psi(states, "sonoff") * (
                    trace[param_name]["on_sonoff"]["dV"]
                    + trace[param_name]["off_sonoff"]["dV"]
                )

                inject_delayed_pre_derivative(
                    args, states, trace, param_name, "on_sonoff", on_pre_deriv, q_before["on_sonoff"]
                )
                inject_delayed_pre_derivative(
                    args, states, trace, param_name, "off_sonoff", off_pre_deriv, q_before["off_sonoff"]
                )
                inject_delayed_pre_derivative(
                    args,
                    states,
                    trace,
                    param_name,
                    "sonoff_ron",
                    sonoff_pre_deriv,
                    q_before["sonoff_ron"],
                )

            states = Eligibility_handler.update_eligibility(args, states, rate_object, timestep)
            states = conditional_handler.run_conditionals(args, states, timestep)

            sonoff_spikers = torch.any(
                states["neurons"]["Dynamic"]["sonoff"]["tspike"] == timestep,
                dim=-1,
            )
            ron_spikers = states["neurons"]["Dynamic"]["ron"]["spikes_holder"][:, :, :, timestep].bool()
            for param_name in PARAM_TO_DERIV:
                if torch.any(sonoff_spikers):
                    trace[param_name]["on_sonoff"]["dV"][sonoff_spikers] = 0
                    trace[param_name]["off_sonoff"]["dV"][sonoff_spikers] = 0
                if torch.any(ron_spikers):
                    trace[param_name]["sonoff_ron"]["dV"][ron_spikers] = 0

    return output


def flatten(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for param_name, param_values in raw.items():
        live = flatten(param_values["live_indirect"])
        stateful = flatten(param_values["stateful_indirect"])
        summary[param_name] = {
            "live_vs_stateful": finite_metrics(live, stateful),
            "neg_live_vs_stateful": finite_metrics(-live, stateful),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check indirect on/off -> sonoff -> ron STRF path.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=3001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "indirect_sonoff_ron_path_check.json",
    )
    args_cli = parser.parse_args()

    with args_cli.config.open("r", encoding="utf-8") as handle:
        base_args = yaml.safe_load(handle)
    test_args = make_test_args(
        base_args,
        cell_target=args_cli.cell,
        batch_size=1,
        sim_len=args_cli.sim_len,
        device=args_cli.device,
    )
    params = select_params_from_config(base_args, test_args, args_cli.cell, args_cli.batch)
    raw = collect(test_args, params, args_cli.seed)
    output = {
        "cell": args_cli.cell,
        "batch": args_cli.batch,
        "sim_len": args_cli.sim_len,
        "seed": args_cli.seed,
        "summary": summarize(raw),
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved results to {args_cli.output}")


if __name__ == "__main__":
    main()
