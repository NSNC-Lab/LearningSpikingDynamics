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
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Pre_Processing import pre_cortical_handler, preprocess_handler
from Simulation import (
    Architecture_Declaration,
    Eligibility_handler,
    Loss_handler,
    conditional_handler,
    ode_handler,
)
from Simulation.initialize_from_mat import (
    get_mat_initialization_config,
    load_last_epoch_params_from_mat,
)


PARAM_TO_STATE = {
    "STRF_gain": ("neurons", "Learnable", "STRF_gain"),
    "STRF_alpha": ("neurons", "Learnable", "STRF_alpha"),
}

PARAM_TO_INIT = {
    "STRF_gain": "Strf_gain",
    "STRF_alpha": "Strf_alpha",
}

PARAM_TO_RATE_DERIV = {
    "STRF_gain": ("onset_rate_gain_deriv", "offset_rate_gain_deriv"),
    "STRF_alpha": ("onset_rate_deriv", "offset_rate_deriv"),
}

REQUIRED_PARAMS = [
    "Strf_gain",
    "Strf_alpha",
    "output_ad",
    "on_ron_gSYN",
    "off_ron_gSYN",
    "on_sonoff_gSYN",
    "off_sonoff_gSYN",
    "sonoff_ron_gSYN",
    "abs_ref",
    "rel_ref_a",
    "rel_ref_b",
    "rel_ref_c",
]


def clone_args(args: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(args)


def make_test_args(
    base_args: dict[str, Any],
    cell_target: int,
    batch_size: int,
    sim_len: int,
    device: str,
) -> dict[str, Any]:
    args = clone_args(base_args)
    args["simulation"]["cell_targets"] = [cell_target]
    args["simulation"]["batch_size"] = batch_size
    args["simulation"]["sim_len"] = sim_len
    args["simulation"]["epochs"] = 1
    args["simulation"]["device"] = device
    args.setdefault("debug_plots", {})["enabled"] = False
    args.setdefault("parameter_initialization", {})["from_mat"] = {"enabled": False}
    return args


def select_params_from_config(
    base_args: dict[str, Any],
    test_args: dict[str, Any],
    cell_target: int,
    batch_index: int,
) -> dict[str, np.ndarray]:
    mat_config = get_mat_initialization_config(base_args)
    if mat_config is None:
        raw_config = base_args.get("parameter_initialization", {}).get("from_mat", {})
        if isinstance(raw_config, dict) and raw_config.get("path"):
            candidate = Path(raw_config["path"])
            if candidate.exists():
                mat_config = dict(raw_config)
                mat_config["enabled"] = True
                base_args = clone_args(base_args)
                base_args["simulation"]["batch_size"] = int(
                    raw_config.get("debug_expected_batches", 12)
                )
                base_args["simulation"]["cell_targets"] = list(
                    range(1, int(raw_config.get("debug_expected_cells", 220)) + 1)
                )
                print(
                    "Using disabled MAT initialization path for finite-difference "
                    f"debugging: {candidate}"
                )
        if mat_config is None:
            raise ValueError(
                "The current config has no enabled MAT initialization and no readable "
                "disabled from_mat.path to use as a debug starting point."
            )

    loaded = load_last_epoch_params_from_mat(mat_config["path"], base_args, mat_config)
    base_cells = list(base_args["simulation"]["cell_targets"])
    if cell_target not in base_cells:
        raise ValueError(f"Cell {cell_target} is not present in the base config.")
    cell_index = base_cells.index(cell_target)

    params: dict[str, np.ndarray] = {}
    for key in REQUIRED_PARAMS:
        arr = np.asarray(loaded[key], dtype=np.float32)
        if batch_index >= arr.shape[0]:
            raise IndexError(f"Batch index {batch_index} is out of bounds for {key} shape {arr.shape}.")
        selected = arr[batch_index : batch_index + 1, cell_index : cell_index + 1]
        if test_args["simulation"]["batch_size"] > 1:
            selected = np.repeat(selected, test_args["simulation"]["batch_size"], axis=0)
        params[key] = np.ascontiguousarray(selected, dtype=np.float32)
    return params


def set_seed(seed: int, device: str) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_gt_data_for_test(args: dict[str, Any]) -> dict[str, torch.Tensor]:
    data_path = REPO_ROOT / args["paths"]["data"]
    mat = loadmat(data_path, variable_names=["all_data"], squeeze_me=True, struct_as_record=False)
    all_data = mat["all_data"]
    device = torch.device(args["simulation"]["device"])
    granularity = int(args["simulation"]["PSTH_granularity"])
    sim_len = int(args["simulation"]["sim_len"])
    bins = sim_len // granularity
    bin_edges = np.arange(bins + 1) * granularity / 10000.0
    max_time_s = sim_len / 10000.0

    psth_holder = torch.zeros(
        (len(args["simulation"]["cell_targets"]), bins), dtype=torch.float32, device=device
    )
    raster_holder = torch.zeros(
        (len(args["simulation"]["cell_targets"]), 10, sim_len), dtype=torch.float32, device=device
    )

    for cell_pos, cell in enumerate(args["simulation"]["cell_targets"]):
        timestamps = all_data[cell - 1].ctrl_tar1_timestamps
        if args["simulation"]["data_target"] != "peak":
            raise NotImplementedError("This debug loader currently mirrors data_target='peak' only.")

        if all_data[cell - 1].tuning_type == "contra-tuned":
            angle = 0
        elif all_data[cell - 1].tuning_type == "45Â°-tuned":
            angle = 1
        elif all_data[cell - 1].tuning_type == "center-tuned":
            angle = 2
        else:
            angle = 3

        timestamps_peak = timestamps[:, angle]
        for trial in range(10):
            trial_timestamps = np.asarray(timestamps_peak[trial])
            if trial_timestamps.size == 0:
                continue
            cut_timesteps = trial_timestamps[
                (trial_timestamps >= 0) & (trial_timestamps < max_time_s)
            ]
            counts, _ = np.histogram(cut_timesteps, bins=bin_edges)
            psth_holder[cell_pos, :] += torch.tensor(counts, dtype=torch.float32, device=device)
            spike_indices = np.floor(cut_timesteps * 10000).astype(np.int64)
            spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < sim_len)]
            raster_holder[cell_pos, trial, spike_indices] = 1

    return {"psth_holder": psth_holder, "raster_holder": raster_holder}


def finite_metrics(analytic: np.ndarray, finite_difference: np.ndarray) -> dict[str, float]:
    analytic = np.asarray(analytic, dtype=np.float64).ravel()
    finite_difference = np.asarray(finite_difference, dtype=np.float64).ravel()
    mask = np.isfinite(analytic) & np.isfinite(finite_difference)
    analytic = analytic[mask]
    finite_difference = finite_difference[mask]

    if analytic.size == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "relative_rmse": float("nan"),
            "correlation": float("nan"),
            "slope_fd_on_analytic": float("nan"),
            "sign_agreement": float("nan"),
        }

    diff = analytic - finite_difference
    rmse = float(np.sqrt(np.mean(diff**2)))
    fd_rms = float(np.sqrt(np.mean(finite_difference**2)))
    denom = float(np.dot(analytic, analytic))
    slope = float(np.dot(analytic, finite_difference) / denom) if denom > 0 else float("nan")

    centered_a = analytic - np.mean(analytic)
    centered_fd = finite_difference - np.mean(finite_difference)
    corr_denom = float(np.sqrt(np.sum(centered_a**2) * np.sum(centered_fd**2)))
    corr = float(np.sum(centered_a * centered_fd) / corr_denom) if corr_denom > 0 else float("nan")

    sign_mask = (np.abs(analytic) + np.abs(finite_difference)) > max(1e-12, 1e-6 * fd_rms)
    sign_agreement = (
        float(np.mean(np.sign(analytic[sign_mask]) == np.sign(finite_difference[sign_mask])))
        if np.any(sign_mask)
        else float("nan")
    )

    return {
        "n": int(analytic.size),
        "rmse": rmse,
        "relative_rmse": float(rmse / (fd_rms + 1e-12)),
        "correlation": corr,
        "slope_fd_on_analytic": slope,
        "sign_agreement": sign_agreement,
        "analytic_rms": float(np.sqrt(np.mean(analytic**2))),
        "finite_difference_rms": fd_rms,
    }


def build_states(args: dict[str, Any], params: dict[str, np.ndarray]) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    return Architecture_Declaration.build_network(args, device, params)


def perturb_params(params: dict[str, np.ndarray], param_name: str, delta: float) -> dict[str, np.ndarray]:
    perturbed = {key: value.copy() for key, value in params.items()}
    perturbed[PARAM_TO_INIT[param_name]][0, 0] += np.float32(delta)
    return perturbed


def rate_derivative_test(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    params_to_test: list[str],
) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    states = build_states(args, params)
    rates = pre_cortical_handler.create_input_fr(args, states, device)

    results: dict[str, Any] = {}
    base_values = {
        "STRF_gain": float(params["Strf_gain"][0, 0]),
        "STRF_alpha": float(params["Strf_alpha"][0, 0]),
    }
    epsilons = {
        "STRF_gain": max(abs(base_values["STRF_gain"]) * 0.01, 1e-5),
        "STRF_alpha": max(abs(base_values["STRF_alpha"]) * 0.01, 0.05),
    }

    for param_name, epsilon in epsilons.items():
        if param_name not in params_to_test:
            continue
        plus_states = build_states(args, perturb_params(params, param_name, epsilon))
        minus_states = build_states(args, perturb_params(params, param_name, -epsilon))
        plus_rates = pre_cortical_handler.create_input_fr(args, plus_states, device)
        minus_rates = pre_cortical_handler.create_input_fr(args, minus_states, device)

        onset_deriv_key, offset_deriv_key = PARAM_TO_RATE_DERIV[param_name]
        onset_fd = (
            tensor_to_numpy(plus_rates["onset_rate"]) - tensor_to_numpy(minus_rates["onset_rate"])
        ) / (2.0 * epsilon)
        offset_fd = (
            tensor_to_numpy(plus_rates["offset_rate"]) - tensor_to_numpy(minus_rates["offset_rate"])
        ) / (2.0 * epsilon)

        results[param_name] = {
            "epsilon": float(epsilon),
            "base_value": base_values[param_name],
            "onset": finite_metrics(tensor_to_numpy(rates[onset_deriv_key]), onset_fd),
            "offset": finite_metrics(tensor_to_numpy(rates[offset_deriv_key]), offset_fd),
        }
    return results


def current_psth_bin(
    states: dict[str, Any],
    gt_data: dict[str, torch.Tensor],
    args: dict[str, Any],
    timestep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    granularity = int(args["simulation"]["PSTH_granularity"])
    holder = states["neurons"]["Dynamic"]["ron"]["spikes_holder"][:, :, :, timestep - granularity : timestep]
    sim_psth = torch.sum(torch.sum(holder, dim=1, keepdim=False), dim=-1, keepdim=False).to(torch.float32)
    data_psth = gt_data["psth_holder"][:, int(timestep / granularity) - 1]
    return sim_psth, data_psth


def run_one_epoch(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    gt_data: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, args["simulation"]["device"])
    states = build_states(args, params)

    with torch.no_grad():
        pre_processed_object = preprocess_handler.preprocess(args, states, device)
        bin_records = []
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(args, states, pre_processed_object["spks"], timestep)
            states = Eligibility_handler.update_eligibility(
                args, states, pre_processed_object["onset_offset_rates"], timestep
            )
            states = conditional_handler.run_conditionals(args, states, timestep)

            if (
                timestep % int(args["simulation"]["PSTH_granularity"]) == 0
                and timestep > 0
            ):
                sim_psth, data_psth = current_psth_bin(states, gt_data, args, timestep)
                gradient = 2.0 * (sim_psth - data_psth[None, :])

                bin_records.append(
                    {
                        "timestep": timestep,
                        "sim_psth": float(sim_psth[0, 0].detach().cpu()),
                        "data_psth": float(data_psth[0].detach().cpu()),
                        "STRF_gain_accum": float(
                            states["neurons"]["Learnable"]["STRF_gain_accum"][0, 0].detach().cpu()
                        ),
                        "STRF_alpha_accum": float(
                            states["neurons"]["Learnable"]["STRF_alpha_accum"][0, 0].detach().cpu()
                        ),
                        "Bk": float(states["neurons"]["Learnable"]["Bk"][0, 0].detach().cpu()),
                    }
                )
                states = Loss_handler.update_grad(states, gradient)

    sim_bins = np.asarray([record["sim_psth"] for record in bin_records], dtype=np.float64)
    data_bins = np.asarray([record["data_psth"] for record in bin_records], dtype=np.float64)
    loss = float(np.sum((sim_bins - data_bins) ** 2))

    return {
        "loss": loss,
        "sim_bins": sim_bins,
        "data_bins": data_bins,
        "bin_records": bin_records,
        "STRF_gain_grad": float(states["neurons"]["Learnable"]["STRF_gain_grad"][0, 0].detach().cpu()),
        "STRF_alpha_grad": float(states["neurons"]["Learnable"]["STRF_alpha_grad"][0, 0].detach().cpu()),
        "mean_Bk": float(np.mean([record["Bk"] for record in bin_records])) if bin_records else float("nan"),
    }


def full_psth_finite_difference_test(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    gt_data: dict[str, torch.Tensor],
    seed: int,
    params_to_test: list[str],
) -> dict[str, Any]:
    baseline = run_one_epoch(args, params, gt_data, seed)

    base_values = {
        "STRF_gain": float(params["Strf_gain"][0, 0]),
        "STRF_alpha": float(params["Strf_alpha"][0, 0]),
    }
    epsilons = {
        "STRF_gain": max(abs(base_values["STRF_gain"]) * 0.05, 5e-5),
        "STRF_alpha": max(abs(base_values["STRF_alpha"]) * 0.05, 0.25),
    }

    results: dict[str, Any] = {
        "baseline_loss": baseline["loss"],
        "baseline_mean_Bk": baseline["mean_Bk"],
        "params": base_values,
        "by_param": {},
    }

    for param_name, epsilon in epsilons.items():
        if param_name not in params_to_test:
            continue
        plus = run_one_epoch(args, perturb_params(params, param_name, epsilon), gt_data, seed)
        minus = run_one_epoch(args, perturb_params(params, param_name, -epsilon), gt_data, seed)

        psth_fd = (plus["sim_bins"] - minus["sim_bins"]) / (2.0 * epsilon)
        loss_fd = (plus["loss"] - minus["loss"]) / (2.0 * epsilon)
        accum_key = f"{param_name}_accum"
        grad_key = f"{param_name}_grad"
        accum = np.asarray([record[accum_key] for record in baseline["bin_records"]], dtype=np.float64)
        bk = np.asarray([record["Bk"] for record in baseline["bin_records"]], dtype=np.float64)
        bk_scaled = accum * bk

        analytic_grad = float(baseline[grad_key])
        results["by_param"][param_name] = {
            "epsilon": float(epsilon),
            "loss_plus": plus["loss"],
            "loss_minus": minus["loss"],
            "loss_fd_grad": float(loss_fd),
            "current_surrogate_grad": analytic_grad,
            "current_times_mean_Bk": float(analytic_grad * baseline["mean_Bk"]),
            "same_sign_current_vs_fd": bool(np.sign(analytic_grad) == np.sign(loss_fd)),
            "same_sign_current_times_Bk_vs_fd": bool(
                np.sign(analytic_grad * baseline["mean_Bk"]) == np.sign(loss_fd)
            ),
            "psth_derivative_vs_current_accum": finite_metrics(accum, psth_fd),
            "psth_derivative_vs_current_accum_times_Bk": finite_metrics(bk_scaled, psth_fd),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finite-difference checks for STRF rate derivatives and the current "
            "STRF eligibility accumulator."
        )
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=10001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--params",
        nargs="+",
        choices=["STRF_gain", "STRF_alpha"],
        default=["STRF_gain", "STRF_alpha"],
        help="Which STRF parameters to finite-difference.",
    )
    parser.add_argument("--skip-rate-test", action="store_true")
    parser.add_argument("--skip-full-test", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "finite_difference_strf_eligibility_results.json",
    )
    args_cli = parser.parse_args()

    with args_cli.config.open("r", encoding="utf-8") as handle:
        base_args = yaml.safe_load(handle)

    if args_cli.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        args_cli.device = "cpu"

    test_args = make_test_args(
        base_args,
        cell_target=args_cli.cell,
        batch_size=args_cli.batch_size,
        sim_len=args_cli.sim_len,
        device=args_cli.device,
    )
    params = select_params_from_config(
        base_args,
        test_args,
        cell_target=args_cli.cell,
        batch_index=args_cli.batch,
    )
    gt_data = load_gt_data_for_test(test_args)

    results = {
        "config": str(args_cli.config),
        "cell": args_cli.cell,
        "batch": args_cli.batch,
        "sim_len": args_cli.sim_len,
        "seed": args_cli.seed,
        "device": args_cli.device,
        "params_tested": args_cli.params,
    }

    if not args_cli.skip_rate_test:
        results["rate_derivative_test"] = rate_derivative_test(
            test_args, params, args_cli.params
        )
    if not args_cli.skip_full_test:
        results["full_psth_finite_difference_test"] = full_psth_finite_difference_test(
            test_args, params, gt_data, args_cli.seed, args_cli.params
        )

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved results to {args_cli.output}")


if __name__ == "__main__":
    main()
