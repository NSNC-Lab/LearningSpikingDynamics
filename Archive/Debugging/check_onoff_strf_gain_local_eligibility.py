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
    perturb_params,
    select_params_from_config,
    set_seed,
)
from Pre_Processing import pre_cortical_handler, preprocess_handler  # noqa: E402
from Simulation import Architecture_Declaration, conditional_handler, ode_handler  # noqa: E402


POP_TO_RATE = {
    "on": ("onset_rate", "onset_rate_gain_deriv"),
    "off": ("offset_rate", "offset_rate_gain_deriv"),
}


def build_states(args: dict[str, Any], params: dict[str, np.ndarray]) -> dict[str, Any]:
    return Architecture_Declaration.build_network(args, torch.device(args["simulation"]["device"]), params)


def flatten(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def collect_local_terms(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])

    states = build_states(args, params)
    pre_processed = preprocess_handler.preprocess(args, states, device)
    base_rates = pre_processed["onset_offset_rates"]

    plus_states = build_states(args, perturb_params(params, "STRF_gain", epsilon))
    minus_states = build_states(args, perturb_params(params, "STRF_gain", -epsilon))
    plus_rates = pre_cortical_handler.create_input_fr(args, plus_states, device)
    minus_rates = pre_cortical_handler.create_input_fr(args, minus_states, device)

    holder: dict[str, dict[str, list[np.ndarray]]] = {
        pop: {
            "finite_difference": [],
            "exact_pre_voltage": [],
            "current_post_voltage_no_bk": [],
            "live_one_plus_bk": [],
        }
        for pop in POP_TO_RATE
    }

    dt = float(args["simulation"]["dt"])
    with torch.no_grad():
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(args, states, pre_processed["spks"], timestep)

            bk = -states["synapses"]["Learnable"]["sonoff_ron"]["gSYN"] * (
                states["neurons"]["Static"]["ron"]["V_thresh"]
                - states["synapses"]["Static"]["sonoff_ron"]["ESYN"]
            )

            for pop_name, (rate_key, deriv_key) in POP_TO_RATE.items():
                static = states["neurons"]["Static"][pop_name]
                dynamic = states["neurons"]["Dynamic"][pop_name]
                v_pre = dynamic["V"][:, :, :, -2]
                v_post = dynamic["V"][:, :, :, -1]
                psi_post = 1.0 - torch.tanh(v_post - static["V_thresh"]) ** 2

                rate_base = base_rates[rate_key][timestep, :, None, :]
                rate_plus = plus_rates[rate_key][timestep, :, None, :]
                rate_minus = minus_rates[rate_key][timestep, :, None, :]
                rate_deriv = base_rates[deriv_key][timestep, :, None, :]

                voltage_coeff_pre = (
                    dt
                    * -static["R"]
                    * static["g_postIC"]
                    * (v_pre - static["E_exc"])
                    / static["tau"]
                )
                voltage_coeff_post = (
                    dt
                    * -static["R"]
                    * static["g_postIC"]
                    * (v_post - static["E_exc"])
                    / static["tau"]
                )

                v_plus_proxy = v_post + voltage_coeff_pre * ((rate_plus - rate_base) * dt / 1000.0)
                v_minus_proxy = v_post + voltage_coeff_pre * ((rate_minus - rate_base) * dt / 1000.0)
                local_fd = (
                    torch.tanh(v_plus_proxy - static["V_thresh"])
                    - torch.tanh(v_minus_proxy - static["V_thresh"])
                ) / (2.0 * epsilon)

                exact_pre_voltage = psi_post * voltage_coeff_pre * (dt / 1000.0) * rate_deriv
                current_post_voltage = psi_post * voltage_coeff_post * (dt / 1000.0) * rate_deriv
                live_one_plus_bk = (1.0 + bk[:, None, :]) * current_post_voltage

                holder[pop_name]["finite_difference"].append(tensor_numpy(torch.sum(local_fd, dim=1)))
                holder[pop_name]["exact_pre_voltage"].append(
                    tensor_numpy(torch.sum(exact_pre_voltage, dim=1))
                )
                holder[pop_name]["current_post_voltage_no_bk"].append(
                    tensor_numpy(torch.sum(current_post_voltage, dim=1))
                )
                holder[pop_name]["live_one_plus_bk"].append(
                    tensor_numpy(torch.sum(live_one_plus_bk, dim=1))
                )

            states = conditional_handler.run_conditionals(args, states, timestep)

    return holder


def summarize(holder: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for pop_name, pop_terms in holder.items():
        fd = flatten(pop_terms["finite_difference"])
        summary[pop_name] = {}
        for term_name in ("exact_pre_voltage", "current_post_voltage_no_bk", "live_one_plus_bk"):
            values = flatten(pop_terms[term_name])
            summary[pop_name][term_name] = finite_metrics(values, fd)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check local on/off STRF_gain eligibility before downstream propagation."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=3001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "onoff_strf_gain_local_eligibility.json",
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
    base_gain = float(params["Strf_gain"][0, 0])
    epsilon = max(abs(base_gain) * 0.01, 1e-5)

    holder = collect_local_terms(test_args, params, args_cli.seed, epsilon)
    output = {
        "cell": args_cli.cell,
        "batch": args_cli.batch,
        "sim_len": args_cli.sim_len,
        "seed": args_cli.seed,
        "epsilon": epsilon,
        "summary": summarize(holder),
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output["summary"], indent=2))
    print(f"Saved results to {args_cli.output}")


if __name__ == "__main__":
    main()
