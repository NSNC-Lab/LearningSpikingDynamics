"""Compare LongRunResults.mat against 60_Epoch_allCells_withRef.mat.

This script recomputes PSTH SSE directly from saved rasters for both runs,
then compares the best batch per cell, CV, and firing rate.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_BASELINE_MAT = REPO_ROOT / "80_epoch_checkpoint_strffix4.mat"
DEFAULT_LONGRUN_MAT = REPO_ROOT / "Archive" / "LongRunResults.mat"
DEFAULT_CONFIG = REPO_ROOT / "simulation_config.yaml"
DEFAULT_OUT_DIR = SCRIPT_DIR / "longrun_vs_80_Epoch_strfFixes4_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and compare PSTH loss, CV, and firing rates."
    )
    parser.add_argument("--baseline-mat", type=Path, default=DEFAULT_BASELINE_MAT)
    parser.add_argument("--longrun-mat", type=Path, default=DEFAULT_LONGRUN_MAT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_matlab_text(value) -> str:
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).ravel())
    return str(value)


def target_angle_for_cell(cell_data) -> int:
    tuning_type = normalize_matlab_text(getattr(cell_data, "tuning_type", ""))
    if tuning_type == "contra-tuned":
        return 0
    if "45" in tuning_type:
        return 1
    if tuning_type == "center-tuned":
        return 2
    return 3


def load_data_raster(
    config: dict, cell_ids: list[int], sim_len: int, dt_ms: float
) -> np.ndarray:
    data_path = REPO_ROOT / config["paths"]["data"]
    mat = loadmat(
        data_path,
        variable_names=["all_data"],
        squeeze_me=True,
        struct_as_record=False,
    )
    all_data = mat["all_data"]
    trials = 10
    raster = np.zeros((len(cell_ids), trials, sim_len), dtype=np.uint8)
    sample_rate = 1000.0 / dt_ms

    for cell_pos, cell_id in enumerate(cell_ids):
        cell_data = all_data[cell_id - 1]
        timestamps = cell_data.ctrl_tar1_timestamps
        angle = target_angle_for_cell(cell_data)
        timestamps_peak = timestamps[:, angle]

        for trial in range(trials):
            trial_times = np.asarray(timestamps_peak[trial]).squeeze()
            if trial_times.size == 0:
                continue
            trial_times = np.atleast_1d(trial_times).astype(float)
            valid = trial_times[
                (trial_times >= 0) & (trial_times < sim_len / sample_rate)
            ]
            spike_indices = (valid * sample_rate).astype(np.int64)
            spike_indices = spike_indices[
                (spike_indices >= 0) & (spike_indices < sim_len)
            ]
            raster[cell_pos, trial, spike_indices] = 1

    return raster


def find_unique_axis(shape: tuple[int, ...], size: int, role: str) -> int:
    matches = [axis for axis, axis_size in enumerate(shape) if axis_size == size]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {role} axis of size {size}, "
            f"found {matches} in shape {shape}."
        )
    return matches[0]


def find_unique_axis_excluding(
    shape: tuple[int, ...], size: int, role: str, excluded_axes: set[int]
) -> int:
    matches = [
        axis
        for axis, axis_size in enumerate(shape)
        if axis_size == size and axis not in excluded_axes
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {role} axis of size {size}, "
            f"found {matches} in shape {shape} after excluding {sorted(excluded_axes)}."
        )
    return matches[0]


def identify_nonbatch_axes(
    shape: tuple[int, ...],
    expected_trials: int,
    expected_cells: int,
    expected_steps: int,
) -> dict[str, int]:
    axes = {"cell": find_unique_axis(shape, expected_cells, "cell")}
    excluded_axes = set(axes.values())
    axes["time"] = find_unique_axis_excluding(
        shape, expected_steps, "time", excluded_axes
    )
    excluded_axes.add(axes["time"])
    axes["trial"] = find_unique_axis_excluding(
        shape, expected_trials, "trial", excluded_axes
    )
    return axes


def identify_batch_axis(
    shape: tuple[int, ...],
    used_axes: set[int],
    expected_batches: int | None,
    run_name: str,
) -> int:
    candidates = [axis for axis in range(len(shape)) if axis not in used_axes]
    non_singleton_candidates = [
        axis for axis in candidates if shape[axis] > 1
    ]

    if expected_batches is not None:
        expected_matches = [
            axis
            for axis in non_singleton_candidates
            if shape[axis] == expected_batches
        ]
        if len(expected_matches) == 1:
            return expected_matches[0]

    if len(non_singleton_candidates) == 1:
        axis = non_singleton_candidates[0]
        print(
            f"{run_name}: inferred batch axis {axis} "
            f"with size {shape[axis]}"
        )
        return axis

    if len(candidates) == 1:
        axis = candidates[0]
        print(
            f"{run_name}: inferred singleton batch axis {axis} "
            f"with size {shape[axis]}"
        )
        return axis

    raise ValueError(
        f"{run_name}: could not identify batch axis in shape {shape}; "
        f"remaining candidate axes are {candidates} with sizes "
        f"{[shape[axis] for axis in candidates]}."
    )


def normalize_output_layout(
    raw_output: np.ndarray,
    run_name: str,
    expected_batches: int | None,
    expected_trials: int,
    expected_cells: int,
    expected_steps: int,
) -> np.ndarray:
    """Return output as batch x trial x cell x time.

    Extra axes are treated as epoch/condition axes and the final index is used.
    This selects the saved last epoch for LongRunResults.mat.
    """
    output = np.asarray(raw_output)

    while output.ndim > 4:
        core_axes = identify_nonbatch_axes(
            output.shape,
            expected_trials,
            expected_cells,
            expected_steps,
        )
        used_axes = set(core_axes.values())
        batch_axis = identify_batch_axis(
            output.shape,
            used_axes,
            expected_batches,
            run_name,
        )
        used_axes.add(batch_axis)
        extra_axes = [axis for axis in range(output.ndim) if axis not in used_axes]
        if not extra_axes:
            raise ValueError(
                f"{run_name}: shape {output.shape} has more than 4 dimensions, "
                "but no non-core axis could be identified."
            )

        singleton_extra_axes = [
            axis for axis in extra_axes if output.shape[axis] == 1
        ]
        axis = singleton_extra_axes[-1] if singleton_extra_axes else extra_axes[-1]
        print(
            f"{run_name}: selecting final index from extra axis {axis} "
            f"with size {output.shape[axis]}"
        )
        output = np.take(output, indices=-1, axis=axis)

    core_axes = identify_nonbatch_axes(
        output.shape,
        expected_trials,
        expected_cells,
        expected_steps,
    )
    core_axes["batch"] = identify_batch_axis(
        output.shape,
        set(core_axes.values()),
        expected_batches,
        run_name,
    )
    normalized = np.transpose(
        output,
        (
            core_axes["batch"],
            core_axes["trial"],
            core_axes["cell"],
            core_axes["time"],
        ),
    )
    return np.ascontiguousarray(normalized, dtype=np.uint8)


def psth_from_sim_output(output: np.ndarray, bin_steps: int) -> np.ndarray:
    batches, trials, cells, steps = output.shape
    bins = steps // bin_steps
    trimmed = output[:, :, :, : bins * bin_steps]
    binned = trimmed.reshape(batches, trials, cells, bins, bin_steps)
    return binned.sum(axis=(1, 4), dtype=np.float64)


def psth_from_data_raster(raster: np.ndarray, bin_steps: int) -> np.ndarray:
    cells, trials, steps = raster.shape
    bins = steps // bin_steps
    trimmed = raster[:, :, : bins * bin_steps]
    binned = trimmed.reshape(cells, trials, bins, bin_steps)
    return binned.sum(axis=(1, 3), dtype=np.float64)


def compute_best_psth_losses(
    output: np.ndarray, data_psth: np.ndarray, bin_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_psth = psth_from_sim_output(output, bin_steps)
    losses = np.sum((sim_psth - data_psth[None, :, :]) ** 2, axis=2)
    best_batches = np.argmin(losses, axis=0)
    best_losses = losses[best_batches, np.arange(losses.shape[1])]
    return best_batches, best_losses, losses


def cv_from_raster(raster: np.ndarray, dt_ms: float) -> float:
    isis = []
    for trial_raster in raster:
        spike_steps = np.flatnonzero(trial_raster)
        if spike_steps.size > 1:
            isis.append(np.diff(spike_steps).astype(np.float64) * dt_ms / 1000.0)

    if not isis:
        return float("nan")

    merged_isis = np.concatenate(isis)
    if merged_isis.size < 2:
        return float("nan")

    mean_isi = float(np.mean(merged_isis))
    if mean_isi == 0.0:
        return float("nan")
    return float(np.std(merged_isis, ddof=1) / mean_isi)


def firing_rate_from_raster(raster: np.ndarray, dt_ms: float) -> float:
    trials, steps = raster.shape
    duration_seconds = steps * dt_ms / 1000.0
    return float(np.sum(raster) / (trials * duration_seconds))


def data_metrics(data_raster: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    cells = data_raster.shape[0]
    cvs = np.empty(cells, dtype=np.float64)
    firing_rates = np.empty(cells, dtype=np.float64)
    for cell_pos in range(cells):
        raster = data_raster[cell_pos, :, :]
        cvs[cell_pos] = cv_from_raster(raster, dt_ms)
        firing_rates[cell_pos] = firing_rate_from_raster(raster, dt_ms)
    return cvs, firing_rates


def selected_sim_metrics(
    output: np.ndarray, best_batches: np.ndarray, dt_ms: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = output.shape[2]
    cvs = np.empty(cells, dtype=np.float64)
    firing_rates = np.empty(cells, dtype=np.float64)
    selected_spikes = np.empty(cells, dtype=np.int64)

    for cell_pos in range(cells):
        batch = int(best_batches[cell_pos])
        raster = output[batch, :, cell_pos, :]
        cvs[cell_pos] = cv_from_raster(raster, dt_ms)
        firing_rates[cell_pos] = firing_rate_from_raster(raster, dt_ms)
        selected_spikes[cell_pos] = int(np.sum(raster))

    return cvs, firing_rates, selected_spikes


def process_run(
    mat_path: Path,
    run_name: str,
    data_psth: np.ndarray,
    bin_steps: int,
    dt_ms: float,
    expected_batches: int,
    expected_trials: int,
    expected_cells: int,
    expected_steps: int,
) -> dict[str, np.ndarray]:
    print(f"Loading {run_name}: {mat_path}")
    mat = loadmat(mat_path, variable_names=["output"], squeeze_me=False)
    output = normalize_output_layout(
        mat["output"],
        run_name,
        expected_batches,
        expected_trials,
        expected_cells,
        expected_steps,
    )
    del mat

    print(f"{run_name}: normalized output shape {output.shape}")
    best_batches, best_losses, losses = compute_best_psth_losses(
        output, data_psth, bin_steps
    )
    cvs, firing_rates, selected_spikes = selected_sim_metrics(
        output, best_batches, dt_ms
    )

    return {
        "best_batches": best_batches,
        "best_losses": best_losses,
        "losses": losses,
        "cvs": cvs,
        "firing_rates": firing_rates,
        "selected_spikes": selected_spikes,
    }


def write_tables(
    out_dir: Path,
    cell_ids: list[int],
    baseline: dict[str, np.ndarray],
    longrun: dict[str, np.ndarray],
    data_cvs: np.ndarray,
    data_firing_rates: np.ndarray,
) -> dict[str, float | int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    delta = longrun["best_losses"] - baseline["best_losses"]
    improved = delta < 0
    worsened = delta > 0
    unchanged = delta == 0

    csv_path = out_dir / "comparison_metrics_by_cell.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cell_id",
                "best_batch_60epoch_zero_based",
                "best_batch_60epoch_one_based",
                "best_psth_sse_60epoch",
                "best_batch_longrun_zero_based",
                "best_batch_longrun_one_based",
                "best_psth_sse_longrun",
                "loss_delta_longrun_minus_60epoch",
                "longrun_is_lower_loss",
                "cv_data",
                "cv_60epoch",
                "cv_longrun",
                "fr_hz_data",
                "fr_hz_60epoch",
                "fr_hz_longrun",
                "spikes_60epoch_best_batch",
                "spikes_longrun_best_batch",
            ]
        )
        for cell_pos, cell_id in enumerate(cell_ids):
            writer.writerow(
                [
                    cell_id,
                    int(baseline["best_batches"][cell_pos]),
                    int(baseline["best_batches"][cell_pos]) + 1,
                    float(baseline["best_losses"][cell_pos]),
                    int(longrun["best_batches"][cell_pos]),
                    int(longrun["best_batches"][cell_pos]) + 1,
                    float(longrun["best_losses"][cell_pos]),
                    float(delta[cell_pos]),
                    bool(improved[cell_pos]),
                    float(data_cvs[cell_pos]),
                    float(baseline["cvs"][cell_pos]),
                    float(longrun["cvs"][cell_pos]),
                    float(data_firing_rates[cell_pos]),
                    float(baseline["firing_rates"][cell_pos]),
                    float(longrun["firing_rates"][cell_pos]),
                    int(baseline["selected_spikes"][cell_pos]),
                    int(longrun["selected_spikes"][cell_pos]),
                ]
            )

    summary = {
        "cells": len(cell_ids),
        "longrun_lower_loss_cells": int(np.sum(improved)),
        "longrun_higher_loss_cells": int(np.sum(worsened)),
        "equal_loss_cells": int(np.sum(unchanged)),
        "mean_best_psth_sse_60epoch": float(np.mean(baseline["best_losses"])),
        "mean_best_psth_sse_longrun": float(np.mean(longrun["best_losses"])),
        "median_best_psth_sse_60epoch": float(np.median(baseline["best_losses"])),
        "median_best_psth_sse_longrun": float(np.median(longrun["best_losses"])),
        "mean_loss_delta_longrun_minus_60epoch": float(np.mean(delta)),
        "median_loss_delta_longrun_minus_60epoch": float(np.median(delta)),
        "mean_cv_data": float(np.nanmean(data_cvs)),
        "mean_cv_60epoch": float(np.nanmean(baseline["cvs"])),
        "mean_cv_longrun": float(np.nanmean(longrun["cvs"])),
        "median_cv_data": float(np.nanmedian(data_cvs)),
        "median_cv_60epoch": float(np.nanmedian(baseline["cvs"])),
        "median_cv_longrun": float(np.nanmedian(longrun["cvs"])),
        "mean_fr_hz_data": float(np.nanmean(data_firing_rates)),
        "mean_fr_hz_60epoch": float(np.nanmean(baseline["firing_rates"])),
        "mean_fr_hz_longrun": float(np.nanmean(longrun["firing_rates"])),
        "median_fr_hz_data": float(np.nanmedian(data_firing_rates)),
        "median_fr_hz_60epoch": float(np.nanmedian(baseline["firing_rates"])),
        "median_fr_hz_longrun": float(np.nanmedian(longrun["firing_rates"])),
    }

    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.save(out_dir / "psth_losses_60epoch_by_batch_cell.npy", baseline["losses"])
    np.save(out_dir / "psth_losses_longrun_by_batch_cell.npy", longrun["losses"])

    print(f"Wrote table: {csv_path}")
    print(f"Wrote summary: {out_dir / 'summary_metrics.json'}")
    return summary


def identity_limits(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([first[np.isfinite(first)], second[np.isfinite(second)]])
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.min(values))
    high = float(np.max(values))
    pad = (high - low) * 0.05 if high > low else max(abs(high) * 0.05, 1.0)
    return low - pad, high + pad


def plot_loss_comparison(
    out_dir: Path,
    cell_ids: np.ndarray,
    baseline_losses: np.ndarray,
    longrun_losses: np.ndarray,
    dpi: int,
) -> plt.Figure:
    delta = longrun_losses - baseline_losses
    colors = np.where(delta < 0, "#15803d", np.where(delta > 0, "#b91c1c", "#6b7280"))
    improved = int(np.sum(delta < 0))
    worsened = int(np.sum(delta > 0))

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)

    axes[0].bar(cell_ids, delta, color=colors, width=0.85)
    axes[0].axhline(0, color="#111827", lw=0.8)
    axes[0].set_title(
        f"Best PSTH SSE delta by cell (LongRun - 60 epoch): "
        f"{improved} lower, {worsened} higher"
    )
    axes[0].set_xlabel("Cell")
    axes[0].set_ylabel("SSE delta")
    axes[0].grid(axis="y", alpha=0.25)

    low, high = identity_limits(baseline_losses, longrun_losses)
    axes[1].scatter(baseline_losses, longrun_losses, c=colors, s=24, alpha=0.85)
    axes[1].plot([low, high], [low, high], color="#111827", lw=1.0)
    axes[1].set_xlim(low, high)
    axes[1].set_ylim(low, high)
    axes[1].set_title("Best PSTH SSE per cell")
    axes[1].set_xlabel("60 epoch")
    axes[1].set_ylabel("LongRun")
    axes[1].grid(alpha=0.25)

    fig.savefig(out_dir / "loss_delta_by_cell.png", dpi=dpi)
    return fig


def plot_cv_comparison(
    out_dir: Path,
    cell_ids: np.ndarray,
    data_cvs: np.ndarray,
    baseline_cvs: np.ndarray,
    longrun_cvs: np.ndarray,
    dpi: int,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)

    axes[0].plot(cell_ids, data_cvs, color="#111827", lw=1.0, label="Data")
    axes[0].plot(cell_ids, baseline_cvs, color="#2563eb", lw=1.0, label="60 epoch")
    axes[0].plot(cell_ids, longrun_cvs, color="#16a34a", lw=1.0, label="LongRun")
    axes[0].set_title("CV by cell, using each run's best PSTH-loss batch")
    axes[0].set_xlabel("Cell")
    axes[0].set_ylabel("ISI CV")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    means = [np.nanmean(data_cvs), np.nanmean(baseline_cvs), np.nanmean(longrun_cvs)]
    medians = [
        np.nanmedian(data_cvs),
        np.nanmedian(baseline_cvs),
        np.nanmedian(longrun_cvs),
    ]
    x = np.arange(3)
    axes[1].bar(x - 0.17, means, width=0.34, color=["#111827", "#2563eb", "#16a34a"])
    axes[1].bar(
        x + 0.17,
        medians,
        width=0.34,
        color=["#6b7280", "#60a5fa", "#86efac"],
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Data", "60 epoch", "LongRun"])
    axes[1].set_ylabel("ISI CV")
    axes[1].set_title("Population average and median CV")
    axes[1].legend(["Mean", "Median"], frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(out_dir / "cv_comparison.png", dpi=dpi)
    return fig


def plot_firing_rate_comparison(
    out_dir: Path,
    cell_ids: np.ndarray,
    data_rates: np.ndarray,
    baseline_rates: np.ndarray,
    longrun_rates: np.ndarray,
    dpi: int,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)

    axes[0].plot(cell_ids, data_rates, color="#111827", lw=1.0, label="Data")
    axes[0].plot(cell_ids, baseline_rates, color="#2563eb", lw=1.0, label="60 epoch")
    axes[0].plot(cell_ids, longrun_rates, color="#16a34a", lw=1.0, label="LongRun")
    axes[0].set_title("Firing rate by cell, using each run's best PSTH-loss batch")
    axes[0].set_xlabel("Cell")
    axes[0].set_ylabel("Hz")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    means = [
        np.nanmean(data_rates),
        np.nanmean(baseline_rates),
        np.nanmean(longrun_rates),
    ]
    medians = [
        np.nanmedian(data_rates),
        np.nanmedian(baseline_rates),
        np.nanmedian(longrun_rates),
    ]
    x = np.arange(3)
    axes[1].bar(x - 0.17, means, width=0.34, color=["#111827", "#2563eb", "#16a34a"])
    axes[1].bar(
        x + 0.17,
        medians,
        width=0.34,
        color=["#6b7280", "#60a5fa", "#86efac"],
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Data", "60 epoch", "LongRun"])
    axes[1].set_ylabel("Hz")
    axes[1].set_title("Population average and median firing rate")
    axes[1].legend(["Mean", "Median"], frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(out_dir / "firing_rate_comparison.png", dpi=dpi)
    return fig


def write_plots(
    out_dir: Path,
    cell_ids: list[int],
    baseline: dict[str, np.ndarray],
    longrun: dict[str, np.ndarray],
    data_cvs: np.ndarray,
    data_firing_rates: np.ndarray,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_array = np.asarray(cell_ids)

    figures = [
        plot_loss_comparison(
            out_dir,
            cell_array,
            baseline["best_losses"],
            longrun["best_losses"],
            dpi,
        ),
        plot_cv_comparison(
            out_dir,
            cell_array,
            data_cvs,
            baseline["cvs"],
            longrun["cvs"],
            dpi,
        ),
        plot_firing_rate_comparison(
            out_dir,
            cell_array,
            data_firing_rates,
            baseline["firing_rates"],
            longrun["firing_rates"],
            dpi,
        ),
    ]

    pdf_path = out_dir / "longrun_vs_60epoch_summary_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote plots PDF: {pdf_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    simulation = config["simulation"]
    cell_ids = [int(cell) for cell in simulation["cell_targets"]]
    dt_ms = float(simulation["dt"])
    bin_steps = int(simulation["PSTH_granularity"])
    expected_batches = int(simulation["batch_size"])
    expected_trials = 10
    expected_cells = len(cell_ids)
    expected_steps = int(simulation["sim_len"])

    print("Building data raster")
    data_raster = load_data_raster(config, cell_ids, expected_steps, dt_ms)
    data_psth = psth_from_data_raster(data_raster, bin_steps)
    data_cvs, data_firing_rates = data_metrics(data_raster, dt_ms)

    baseline = process_run(
        args.baseline_mat,
        "60 epoch",
        data_psth,
        bin_steps,
        dt_ms,
        expected_batches,
        expected_trials,
        expected_cells,
        expected_steps,
    )
    gc.collect()

    longrun = process_run(
        args.longrun_mat,
        "LongRun",
        data_psth,
        bin_steps,
        dt_ms,
        expected_batches,
        expected_trials,
        expected_cells,
        expected_steps,
    )
    gc.collect()

    summary = write_tables(
        args.out_dir,
        cell_ids,
        baseline,
        longrun,
        data_cvs,
        data_firing_rates,
    )
    write_plots(
        args.out_dir,
        cell_ids,
        baseline,
        longrun,
        data_cvs,
        data_firing_rates,
        args.dpi,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
