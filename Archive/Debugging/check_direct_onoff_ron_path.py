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

from finite_difference_strf_eligibility import (  # noqa: E402
    finite_metrics,
    make_test_args,
    select_params_from_config,
    set_seed,
)
from Pre_Processing import preprocess_handler  # noqa: E402
from Simulation import Architecture_Declaration, Eligibility_handler, conditional_handler, ode_handler  # noqa: E402


POP_TO_SYNAPSE = {"on": "on_ron", "off": "off_ron"}
PARAM_TO_DERIV = {
    "STRF_gain": {"on": "onset_rate_gain_deriv", "off": "offset_rate_gain_deriv"},
    "STRF_alpha": {"on": "onset_rate_deriv", "off": "offset_rate_deriv"},
}


def build_states(args: dict[str, Any], params: dict[str, np.ndarray]) -> dict[str, Any]:
    return Architecture_Declaration.build_network(args, torch.device(args["simulation"]["device"]), params)


def zeros_trace(args: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.zeros(
        (
            int(args["simulation"]["batch_size"]),
            10,
            len(args["simulation"]["cell_targets"]),
        ),
        dtype=torch.float32,
        device=device,
    )


def neuron_psi(states: dict[str, Any], neuron_name: str) -> torch.Tensor:
    static = states["neurons"]["Static"][neuron_name]
    dynamic = states["neurons"]["Dynamic"][neuron_name]
    return 1.0 - torch.tanh(dynamic["V"][:, :, :, -1] - static["V_thresh"]) ** 2


def delayed(trace: dict[str, Any], key: str, value: torch.Tensor, delay_steps: int) -> torch.Tensor:
    if delay_steps == 0:
        return value
    if key not in trace["buffers"]:
        trace["buffers"][key] = torch.zeros(
            (delay_steps, *value.shape), device=value.device, dtype=torch.float32
        )
    output = trace["buffers"][key][0].clone()
    trace["buffers"][key][:-1] = trace["buffers"][key][1:].clone()
    trace["buffers"][key][-1] = value
    return output


def init_trace(args: dict[str, Any], device: torch.device) -> dict[str, Any]:
    trace: dict[str, Any] = {"buffers": {}}
    for param_name in PARAM_TO_DERIV:
        trace[param_name] = {}
        for pop_name, synapse_name in POP_TO_SYNAPSE.items():
            trace[param_name][pop_name] = {
                "dPSC_s": zeros_trace(args, device),
                "dPSC_x": zeros_trace(args, device),
                "dV_ron": zeros_trace(args, device),
            }
    return trace


def local_pre_spike_derivative(
    args: dict[str, Any],
    states: dict[str, Any],
    rate_object: dict[str, torch.Tensor],
    timestep: int,
    param_name: str,
    pop_name: str,
) -> torch.Tensor:
    static = states["neurons"]["Static"][pop_name]
    dynamic = states["neurons"]["Dynamic"][pop_name]
    dt = float(args["simulation"]["dt"])
    deriv_key = PARAM_TO_DERIV[param_name][pop_name]
    rate_deriv = rate_object[deriv_key][timestep, :, None, :]
    v_pre = dynamic["V"][:, :, :, -2]
    psi_post = neuron_psi(states, pop_name)
    d_v_pre = (
        dt
        * -static["R"]
        * static["g_postIC"]
        * (v_pre - static["E_exc"])
        / static["tau"]
        * (dt / 1000.0)
        * rate_deriv
    )
    return psi_post * d_v_pre


def live_direct_path_piece(
    args: dict[str, Any],
    states: dict[str, Any],
    rate_object: dict[str, torch.Tensor],
    timestep: int,
    param_name: str,
    pop_name: str,
) -> torch.Tensor:
    synapse_name = POP_TO_SYNAPSE[pop_name]
    syn_dynamic = states["synapses"]["Dynamic"][synapse_name]
    param_short = "gain" if param_name == "STRF_gain" else "alpha"
    return torch.sum(neuron_psi(states, "ron") * syn_dynamic[f"dV_{param_short}"][:, :, :, -1], dim=1)


def advance_direct_sensitivity(
    args: dict[str, Any],
    states: dict[str, Any],
    trace: dict[str, Any],
    pre_deriv: torch.Tensor,
    q_before: torch.Tensor,
    timestep: int,
    param_name: str,
    pop_name: str,
) -> torch.Tensor:
    synapse_name = POP_TO_SYNAPSE[pop_name]
    syn_static = states["synapses"]["Static"][synapse_name]
    syn_learnable = states["synapses"]["Learnable"][synapse_name]
    ron_static = states["neurons"]["Static"]["ron"]
    ron_dynamic = states["neurons"]["Dynamic"]["ron"]
    ron_psi = neuron_psi(states, "ron")
    dt = float(args["simulation"]["dt"])

    cur_trace = trace[param_name][pop_name]
    old_dpsc_s = cur_trace["dPSC_s"].clone()
    old_dpsc_x = cur_trace["dPSC_x"].clone()
    old_dv_ron = cur_trace["dV_ron"].clone()

    v_old = ron_dynamic["V"][:, :, :, -2]
    g_ad_old = ron_dynamic["g_ad"][:, :, :, -2]
    ron_dv_dv = (-(1.0 + ron_static["R"] * g_ad_old) / ron_static["tau"])
    for projection_name in states["neurons"]["Static"]["ron"]["projections"]:
        projection_psc = states["synapses"]["Dynamic"][projection_name]["PSC_s"][:, :, :, -2]
        projection_gsyn = states["synapses"]["Learnable"][projection_name]["gSYN"][:, None, :]
        ron_dv_dv = ron_dv_dv - ron_static["R"] * projection_psc * projection_gsyn / ron_static["tau"]
    if ron_static["noise"] == 1:
        noise_sn = ron_dynamic["noise_sn"][:, :, :, -2]
        ron_dv_dv = ron_dv_dv - ron_static["R"] * ron_static["nSYN"] * noise_sn / ron_static["tau"]

    d_v_rhs = ron_dv_dv * old_dv_ron
    d_v_rhs = d_v_rhs - (
        ron_static["R"]
        * syn_learnable["gSYN"][:, None, :]
        * old_dpsc_s
        * (v_old - syn_static["ESYN"])
        / ron_static["tau"]
    )

    cur_trace["dV_ron"] = old_dv_ron + dt * d_v_rhs
    cur_trace["dPSC_s"] = old_dpsc_s + dt * (
        syn_static["scale"] * old_dpsc_x - old_dpsc_s
    ) / syn_static["tauR"]
    cur_trace["dPSC_x"] = old_dpsc_x + dt * (-old_dpsc_x / syn_static["tauD"])

    output_deriv = torch.sum(ron_psi * cur_trace["dV_ron"], dim=1)

    delay_steps = int(round(float(syn_static["PSC_delay"]) / dt))
    delayed_pre = delayed(
        trace,
        f"{param_name}_{synapse_name}",
        pre_deriv,
        delay_steps,
    )
    cur_trace["dPSC_x"] = cur_trace["dPSC_x"] + q_before * delayed_pre

    return output_deriv


def collect(args: dict[str, Any], params: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = build_states(args, params)
    trace = init_trace(args, device)
    output = {
        param_name: {
            pop_name: {"stateful_direct": [], "live_direct": []}
            for pop_name in POP_TO_SYNAPSE
        }
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
                for synapse_name in POP_TO_SYNAPSE.values()
            }

            for param_name in PARAM_TO_DERIV:
                for pop_name, synapse_name in POP_TO_SYNAPSE.items():
                    pre_deriv = local_pre_spike_derivative(
                        args, states, rate_object, timestep, param_name, pop_name
                    )
                    stateful = advance_direct_sensitivity(
                        args,
                        states,
                        trace,
                        pre_deriv,
                        q_before[synapse_name],
                        timestep,
                        param_name,
                        pop_name,
                    )
                    live = live_direct_path_piece(
                        args, states, rate_object, timestep, param_name, pop_name
                    )
                    output[param_name][pop_name]["stateful_direct"].append(
                        stateful.detach().cpu().numpy()
                    )
                    output[param_name][pop_name]["live_direct"].append(
                        live.detach().cpu().numpy()
                    )

            states = Eligibility_handler.update_eligibility(args, states, rate_object, timestep)
            states = conditional_handler.run_conditionals(args, states, timestep)
            ron_spikers = states["neurons"]["Dynamic"]["ron"]["spikes_holder"][:, :, :, timestep].bool()
            if torch.any(ron_spikers):
                for param_name in PARAM_TO_DERIV:
                    for pop_name in POP_TO_SYNAPSE:
                        trace[param_name][pop_name]["dV_ron"][ron_spikers] = 0

    return output


def flatten(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for param_name, param_values in raw.items():
        summary[param_name] = {}
        combined_stateful = []
        combined_live = []
        for pop_name, pop_values in param_values.items():
            stateful = flatten(pop_values["stateful_direct"])
            live = flatten(pop_values["live_direct"])
            combined_stateful.append(stateful)
            combined_live.append(live)
            summary[param_name][pop_name] = {
                "live_vs_stateful": finite_metrics(live, stateful),
                "neg_live_vs_stateful": finite_metrics(-live, stateful),
                "dt_live_vs_stateful": finite_metrics(0.1 * live, stateful),
            }
        summary[param_name]["combined_on_plus_off"] = {
            "live_vs_stateful": finite_metrics(
                np.sum(np.vstack(combined_live), axis=0),
                np.sum(np.vstack(combined_stateful), axis=0),
            ),
            "neg_live_vs_stateful": finite_metrics(
                -np.sum(np.vstack(combined_live), axis=0),
                np.sum(np.vstack(combined_stateful), axis=0),
            ),
            "dt_live_vs_stateful": finite_metrics(
                0.1 * np.sum(np.vstack(combined_live), axis=0),
                np.sum(np.vstack(combined_stateful), axis=0),
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check direct on/off -> ron STRF path eligibility.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=3001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "direct_onoff_ron_path_check.json",
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
