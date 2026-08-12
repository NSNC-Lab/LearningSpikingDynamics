"""Evaluate a MAT checkpoint with forward dynamics only.

This script intentionally does not call the eligibility, loss-gradient, or
Adam handlers.  It rebuilds a selected checkpoint slice for every requested
seed, runs the production preprocessing/ODE/conditional path, and reports
firing-rate and aligned PSTH metrics as JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data import data_handler  # noqa: E402
from Pre_Processing import preprocess_handler  # noqa: E402
from Simulation import Architecture_Declaration, conditional_handler, ode_handler  # noqa: E402
from Simulation.initialize_from_mat import (  # noqa: E402
    load_last_epoch_params_from_mat,
    resolve_mat_path,
)


REQUIRED_PARAMS = (
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
)


def set_seed(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def validate_unique(values: list[int], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates: {values}")


def select_checkpoint_params(
    loaded: dict[str, np.ndarray],
    batch_indices: list[int],
    checkpoint_cells: list[int],
    selected_cells: list[int],
) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_PARAMS if name not in loaded]
    if missing:
        raise KeyError(
            "Checkpoint does not contain all parameters required by the current "
            f"network: {missing}"
        )

    cell_positions = [checkpoint_cells.index(cell) for cell in selected_cells]
    selected: dict[str, np.ndarray] = {}
    for name in REQUIRED_PARAMS:
        values = np.asarray(loaded[name], dtype=np.float32)
        selected[name] = np.ascontiguousarray(
            values[np.ix_(batch_indices, cell_positions)], dtype=np.float32
        )
    return selected


def correlation_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def weighted_time_or_none(counts: np.ndarray, centers_ms: np.ndarray) -> float | None:
    total = float(np.sum(counts))
    if total <= 0:
        return None
    return float(np.dot(counts, centers_ms) / total)


def summarize_forward(
    spikes: np.ndarray,
    target_psth: np.ndarray,
    args: dict[str, Any],
    checkpoint_batches: list[int],
    cells: list[int],
    include_psth_arrays: bool,
) -> dict[str, Any]:
    dt_ms = float(args["simulation"]["dt"])
    sim_len = int(args["simulation"]["sim_len"])
    granularity = int(args["simulation"]["PSTH_granularity"])
    complete_bins = sim_len // granularity
    covered_steps = complete_bins * granularity
    full_duration_s = sim_len * dt_ms / 1000.0
    covered_duration_s = covered_steps * dt_ms / 1000.0

    if complete_bins < 1:
        raise ValueError(
            f"sim_len={sim_len} is shorter than one PSTH bin of {granularity} steps."
        )

    batch_count, trial_count, cell_count, _ = spikes.shape
    sim_psth = spikes[..., :covered_steps].reshape(
        batch_count, trial_count, cell_count, complete_bins, granularity
    ).sum(axis=(1, 4))
    target_psth = np.asarray(target_psth[:, :complete_bins], dtype=np.float64)
    residual = sim_psth.astype(np.float64) - target_psth[None, :, :]

    full_spike_counts = spikes.sum(axis=(1, 3))
    binned_spike_counts = sim_psth.sum(axis=-1)
    target_spike_counts = target_psth.sum(axis=-1)
    bin_centers_ms = (np.arange(complete_bins, dtype=np.float64) + 0.5) * granularity * dt_ms

    records: list[dict[str, Any]] = []
    for local_batch, checkpoint_batch in enumerate(checkpoint_batches):
        for cell_pos, cell in enumerate(cells):
            sim_curve = sim_psth[local_batch, cell_pos].astype(np.float64)
            target_curve = target_psth[cell_pos]
            sim_has_spikes = bool(np.sum(sim_curve) > 0)
            target_has_spikes = bool(np.sum(target_curve) > 0)
            sim_peak = int(np.argmax(sim_curve)) if sim_has_spikes else None
            target_peak = int(np.argmax(target_curve)) if target_has_spikes else None
            sim_peak_time_ms = (
                float(bin_centers_ms[sim_peak]) if sim_peak is not None else None
            )
            target_peak_time_ms = (
                float(bin_centers_ms[target_peak]) if target_peak is not None else None
            )
            peak_time_error_ms = (
                sim_peak_time_ms - target_peak_time_ms
                if sim_peak_time_ms is not None and target_peak_time_ms is not None
                else None
            )
            squared_error = residual[local_batch, cell_pos] ** 2
            record = {
                "checkpoint_batch": checkpoint_batch,
                "cell": cell,
                "full_spike_count": int(full_spike_counts[local_batch, cell_pos]),
                "full_firing_rate_hz": float(
                    full_spike_counts[local_batch, cell_pos]
                    / (trial_count * full_duration_s)
                ),
                "binned_spike_count": int(binned_spike_counts[local_batch, cell_pos]),
                "binned_firing_rate_hz": float(
                    binned_spike_counts[local_batch, cell_pos]
                    / (trial_count * covered_duration_s)
                ),
                "target_binned_spike_count": float(target_spike_counts[cell_pos]),
                "target_binned_firing_rate_hz": float(
                    target_spike_counts[cell_pos]
                    / (trial_count * covered_duration_s)
                ),
                "psth_sse_sum": float(np.sum(squared_error)),
                "psth_sse_mean_per_bin": float(np.mean(squared_error)),
                "psth_rmse_per_bin": float(np.sqrt(np.mean(squared_error))),
                "psth_correlation": correlation_or_none(sim_curve, target_curve),
                "sim_peak_bin": sim_peak,
                "target_peak_bin": target_peak,
                "sim_peak_time_ms": sim_peak_time_ms,
                "target_peak_time_ms": target_peak_time_ms,
                "peak_time_error_ms": peak_time_error_ms,
                "sim_weighted_time_ms": weighted_time_or_none(sim_curve, bin_centers_ms),
                "target_weighted_time_ms": weighted_time_or_none(target_curve, bin_centers_ms),
            }
            records.append(record)

    result: dict[str, Any] = {
        "aggregate": {
            "full_firing_rate_hz": float(
                full_spike_counts.sum()
                / (batch_count * cell_count * trial_count * full_duration_s)
            ),
            "binned_firing_rate_hz": float(
                binned_spike_counts.sum()
                / (batch_count * cell_count * trial_count * covered_duration_s)
            ),
            "target_binned_firing_rate_hz": float(
                target_spike_counts.sum()
                / (cell_count * trial_count * covered_duration_s)
            ),
            "psth_sse_sum": float(np.sum(residual ** 2)),
            "psth_sse_mean": float(np.mean(residual ** 2)),
            "psth_rmse": float(np.sqrt(np.mean(residual ** 2))),
        },
        "by_batch_cell": records,
    }
    if include_psth_arrays:
        result["sim_psth_counts"] = sim_psth.tolist()
    return result


def rate_summary(rate_object: dict[str, torch.Tensor], sim_len: int) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name in ("onset_rate", "offset_rate"):
        values = rate_object[name][:sim_len].detach().cpu().numpy()
        summary[f"{name}_mean_hz"] = float(np.mean(values))
        summary[f"{name}_max_hz"] = float(np.max(values))
    return summary


def run_seed(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    gt_data: dict[str, torch.Tensor],
    seed: int,
    checkpoint_batches: list[int],
    cells: list[int],
    include_psth_arrays: bool,
) -> dict[str, Any]:
    device = torch.device(args["simulation"]["device"])
    set_seed(seed, device)
    states = Architecture_Declaration.build_network(args, device, params)

    start = time.perf_counter()
    with torch.no_grad():
        preprocessed = preprocess_handler.preprocess(args, states, device)
        for timestep in range(int(args["simulation"]["sim_len"])):
            states = ode_handler.run_odes(
                args, states, preprocessed["spks"], timestep
            )
            states = conditional_handler.run_conditionals(args, states, timestep)
    elapsed = time.perf_counter() - start

    spikes = (
        states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
    )
    result = summarize_forward(
        spikes,
        gt_data["psth_holder"].detach().cpu().numpy(),
        args,
        checkpoint_batches,
        cells,
        include_psth_arrays,
    )
    result.update(
        {
            "seed": seed,
            "runtime_seconds": elapsed,
            "input_rates": rate_summary(
                preprocessed["onset_offset_rates"],
                int(args["simulation"]["sim_len"]),
            ),
        }
    )
    return result


def aggregate_seeds(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_keys = (
        "full_firing_rate_hz",
        "binned_firing_rate_hz",
        "psth_sse_sum",
        "psth_sse_mean",
        "psth_rmse",
    )
    aggregate: dict[str, Any] = {}
    for key in aggregate_keys:
        values = np.asarray(
            [result["aggregate"][key] for result in seed_results], dtype=np.float64
        )
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))

    by_batch_cell: list[dict[str, Any]] = []
    record_count = len(seed_results[0]["by_batch_cell"])
    for record_index in range(record_count):
        records = [result["by_batch_cell"][record_index] for result in seed_results]
        entry: dict[str, Any] = {
            "checkpoint_batch": records[0]["checkpoint_batch"],
            "cell": records[0]["cell"],
        }
        for key in ("full_firing_rate_hz", "psth_sse_sum", "peak_time_error_ms"):
            valid_values = [record[key] for record in records if record[key] is not None]
            if valid_values:
                values = np.asarray(valid_values, dtype=np.float64)
                entry[f"{key}_mean"] = float(np.mean(values))
                entry[f"{key}_std"] = float(np.std(values))
            else:
                entry[f"{key}_mean"] = None
                entry[f"{key}_std"] = None
            if key == "peak_time_error_ms":
                entry["peak_time_error_ms_valid_seed_count"] = len(valid_values)
        by_batch_cell.append(entry)
    aggregate["by_batch_cell"] = by_batch_cell
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward-only evaluation of selected MAT checkpoint batches/cells."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "simulation_config.yaml"
    )
    parser.add_argument("--cells", nargs="+", type=int, required=True)
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[0],
        help="Zero-based checkpoint batch indices (default: 0).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim-len", type=int, default=None)
    parser.add_argument("--psth-granularity", type=int, default=None)
    parser.add_argument(
        "--checkpoint-batch-count",
        type=int,
        default=None,
        help="Checkpoint batch-axis size; defaults to simulation.batch_size in config.",
    )
    parser.add_argument(
        "--checkpoint-cell-targets",
        nargs="+",
        type=int,
        default=None,
        help="Cell IDs represented by the checkpoint cell axis; defaults to config cell_targets.",
    )
    parser.add_argument("--include-psth-arrays", action="store_true")
    parser.add_argument(
        "--output",
        default="-",
        help="JSON output path, or '-' for stdout (default).",
    )
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    config_path = cli.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        base_args = yaml.safe_load(handle)

    validate_unique(cli.cells, "--cells")
    validate_unique(cli.batches, "--batches")
    validate_unique(cli.seeds, "--seeds")

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {cli.device!r}, but CUDA is unavailable.")

    checkpoint_cells = list(
        cli.checkpoint_cell_targets
        if cli.checkpoint_cell_targets is not None
        else base_args["simulation"]["cell_targets"]
    )
    checkpoint_batch_count = int(
        cli.checkpoint_batch_count
        if cli.checkpoint_batch_count is not None
        else base_args["simulation"]["batch_size"]
    )
    validate_unique(checkpoint_cells, "checkpoint cell targets")

    missing_cells = [cell for cell in cli.cells if cell not in checkpoint_cells]
    if missing_cells:
        raise ValueError(
            f"Selected cells {missing_cells} are absent from the checkpoint cell mapping."
        )
    invalid_batches = [
        batch for batch in cli.batches
        if batch < 0 or batch >= checkpoint_batch_count
    ]
    if invalid_batches:
        raise IndexError(
            f"Checkpoint batch indices out of range [0, {checkpoint_batch_count}): "
            f"{invalid_batches}"
        )

    checkpoint_args = copy.deepcopy(base_args)
    checkpoint_args["simulation"]["batch_size"] = checkpoint_batch_count
    checkpoint_args["simulation"]["cell_targets"] = checkpoint_cells
    loader_config = copy.deepcopy(
        base_args.get("parameter_initialization", {}).get("from_mat", {})
    )
    loader_config["enabled"] = True
    loader_config["path"] = str(cli.checkpoint)
    loaded = load_last_epoch_params_from_mat(
        cli.checkpoint, checkpoint_args, loader_config
    )
    selected_params = select_checkpoint_params(
        loaded, cli.batches, checkpoint_cells, cli.cells
    )

    eval_args = copy.deepcopy(base_args)
    eval_args["simulation"]["batch_size"] = len(cli.batches)
    eval_args["simulation"]["cell_targets"] = list(cli.cells)
    eval_args["simulation"]["device"] = cli.device
    eval_args["simulation"]["epochs"] = 1
    if cli.sim_len is not None:
        eval_args["simulation"]["sim_len"] = cli.sim_len
    if cli.psth_granularity is not None:
        eval_args["simulation"]["PSTH_granularity"] = cli.psth_granularity
    eval_args.setdefault("debug_plots", {})["enabled"] = False
    eval_args.setdefault("parameter_initialization", {})["from_mat"] = {
        "enabled": False
    }

    gt_data = data_handler.load_gt_data(eval_args)
    seed_results: list[dict[str, Any]] = []
    for seed in cli.seeds:
        print(f"Evaluating seed {seed}...", file=sys.stderr, flush=True)
        seed_results.append(
            run_seed(
                eval_args,
                selected_params,
                gt_data,
                seed,
                cli.batches,
                cli.cells,
                cli.include_psth_arrays,
            )
        )

    granularity = int(eval_args["simulation"]["PSTH_granularity"])
    dt_ms = float(eval_args["simulation"]["dt"])
    complete_bins = int(eval_args["simulation"]["sim_len"]) // granularity
    payload: dict[str, Any] = {
        "mode": "forward_only_no_eligibility_no_adam",
        "checkpoint": str(resolve_mat_path(cli.checkpoint)),
        "config": str(config_path),
        "device": cli.device,
        "seeds": cli.seeds,
        "selected_checkpoint_batches": cli.batches,
        "selected_cells": cli.cells,
        "simulation": {
            "dt_ms": dt_ms,
            "sim_len_steps": int(eval_args["simulation"]["sim_len"]),
            "trials": 10,
            "psth_granularity_steps": granularity,
            "psth_bin_width_ms": granularity * dt_ms,
            "psth_complete_bins": complete_bins,
            "psth_covered_steps": complete_bins * granularity,
            "psth_interval_convention": "[bin_start_step, bin_end_step)",
            "psth_bin_start_times_ms": (
                np.arange(complete_bins, dtype=np.float64) * granularity * dt_ms
            ).tolist(),
            "psth_bin_end_times_ms": (
                (np.arange(complete_bins, dtype=np.float64) + 1.0)
                * granularity
                * dt_ms
            ).tolist(),
        },
        "parameters": {
            name: values.astype(np.float64).tolist()
            for name, values in selected_params.items()
        },
        "target_psth_counts": (
            gt_data["psth_holder"].detach().cpu().numpy().tolist()
            if cli.include_psth_arrays
            else None
        ),
        "seed_results": seed_results,
        "across_seed_summary": aggregate_seeds(seed_results),
    }

    json_text = json.dumps(payload, indent=2, allow_nan=False)
    if cli.output == "-":
        print(json_text)
    else:
        output_path = Path(cli.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text + "\n", encoding="utf-8")
        print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
