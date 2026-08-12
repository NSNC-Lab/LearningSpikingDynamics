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

from check_direct_onoff_ron_path import local_pre_spike_derivative, neuron_psi, zeros_trace  # noqa: E402
from check_indirect_sonoff_ron_path import (  # noqa: E402
    advance_trace_dynamics,
    inject_delayed_pre_derivative,
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
PARAM_TO_ACCUM = {"STRF_gain": "STRF_gain_accum", "STRF_alpha": "STRF_alpha_accum"}
ALL_PATH_SYNAPSES = ("on_ron", "off_ron", "on_sonoff", "off_sonoff", "sonoff_ron")
POST_BY_SYNAPSE = {
    "on_ron": "ron",
    "off_ron": "ron",
    "sonoff_ron": "ron",
    "on_sonoff": "sonoff",
    "off_sonoff": "sonoff",
}


def build_states(args: dict[str, Any], params: dict[str, np.ndarray]) -> dict[str, Any]:
    return Architecture_Declaration.build_network(args, torch.device(args["simulation"]["device"]), params)


def init_trace(args: dict[str, Any], device: torch.device) -> dict[str, Any]:
    trace: dict[str, Any] = {"buffers": {}}
    for param_name in PARAM_TO_DERIV:
        trace[param_name] = {}
        for synapse_name in ALL_PATH_SYNAPSES:
            trace[param_name][synapse_name] = {
                "dPSC_s": zeros_trace(args, device),
                "dPSC_x": zeros_trace(args, device),
                "dV": zeros_trace(args, device),
            }
    return trace


def full_output_derivative_from_trace(
    states: dict[str, Any],
    trace: dict[str, Any],
    param_name: str,
) -> torch.Tensor:
    return torch.sum(
        neuron_psi(states, "ron")
        * (
            trace[param_name]["on_ron"]["dV"]
            + trace[param_name]["off_ron"]["dV"]
            + trace[param_name]["sonoff_ron"]["dV"]
        ),
        dim=1,
    )


def full_output_derivative_from_live(
    states: dict[str, Any],
    param_name: str,
) -> torch.Tensor:
    param_short = PARAM_TO_SHORT[param_name]
    return torch.sum(
        neuron_psi(states, "ron")
        * (
            states["synapses"]["Dynamic"]["on_ron"][f"dV_{param_short}"][:, :, :, -1]
            + states["synapses"]["Dynamic"]["off_ron"][f"dV_{param_short}"][:, :, :, -1]
            + states["synapses"]["Dynamic"]["sonoff_ron"][f"dV_{param_short}"][:, :, :, -1]
        ),
        dim=1,
    )


def flatten(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])


def collect(args: dict[str, Any], params: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = build_states(args, params)
    trace = init_trace(args, device)
    output = {
        param_name: {
            "stateful_full": [],
            "live_full": [],
            "accum_delta": [],
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
                for synapse_name in ALL_PATH_SYNAPSES
            }
            accum_before = {
                param_name: states["neurons"]["Learnable"][PARAM_TO_ACCUM[param_name]].clone()
                for param_name in PARAM_TO_DERIV
            }

            for param_name in PARAM_TO_DERIV:
                for synapse_name in ALL_PATH_SYNAPSES:
                    advance_trace_dynamics(
                        args,
                        states,
                        trace,
                        param_name,
                        synapse_name,
                        POST_BY_SYNAPSE[synapse_name],
                    )

                output[param_name]["stateful_full"].append(
                    full_output_derivative_from_trace(states, trace, param_name).detach().cpu().numpy()
                )
                output[param_name]["live_full"].append(
                    full_output_derivative_from_live(states, param_name).detach().cpu().numpy()
                )

            states = Eligibility_handler.update_eligibility(args, states, rate_object, timestep)

            for param_name in PARAM_TO_DERIV:
                accum_after = states["neurons"]["Learnable"][PARAM_TO_ACCUM[param_name]]
                output[param_name]["accum_delta"].append(
                    (accum_after - accum_before[param_name]).detach().cpu().numpy()
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
                    args, states, trace, param_name, "on_ron", on_pre_deriv, q_before["on_ron"]
                )
                inject_delayed_pre_derivative(
                    args, states, trace, param_name, "off_ron", off_pre_deriv, q_before["off_ron"]
                )
                inject_delayed_pre_derivative(
                    args,
                    states,
                    trace,
                    param_name,
                    "on_sonoff",
                    on_pre_deriv,
                    q_before["on_sonoff"],
                )
                inject_delayed_pre_derivative(
                    args,
                    states,
                    trace,
                    param_name,
                    "off_sonoff",
                    off_pre_deriv,
                    q_before["off_sonoff"],
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
                    trace[param_name]["on_ron"]["dV"][ron_spikers] = 0
                    trace[param_name]["off_ron"]["dV"][ron_spikers] = 0
                    trace[param_name]["sonoff_ron"]["dV"][ron_spikers] = 0

    return output


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for param_name, values in raw.items():
        stateful = flatten(values["stateful_full"])
        live = flatten(values["live_full"])
        accum_delta = flatten(values["accum_delta"])
        summary[param_name] = {
            "live_full_vs_stateful_full": finite_metrics(live, stateful),
            "accum_delta_vs_stateful_full": finite_metrics(accum_delta, stateful),
            "accum_delta_vs_live_full": finite_metrics(accum_delta, live),
            "neg_accum_delta_vs_stateful_full": finite_metrics(-accum_delta, stateful),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check full d output surrogate spike / d STRF parameter.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=3001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "full_output_strf_derivative_check.json",
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
