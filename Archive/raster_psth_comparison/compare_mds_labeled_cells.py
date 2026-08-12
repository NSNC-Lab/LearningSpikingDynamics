"""Compare LongRun vs 60 epoch for MDS-labeled medium/good/great cells.

The medium/good/great cell IDs are parsed from the MATLAB MDS plotting script,
then PSTH SSE, CV, and firing rates are recomputed from raw rasters.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from compare_longrun_vs_60epoch import (
    DEFAULT_BASELINE_MAT,
    DEFAULT_CONFIG,
    DEFAULT_LONGRUN_MAT,
    REPO_ROOT,
    data_metrics,
    load_config,
    load_data_raster,
    process_run,
    psth_from_data_raster,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MDS_SCRIPT = Path(
    r"C:\Users\ipboy\Desktop\Research\Misc\Backup Figures"
    r"\SingleChannelFigures\Figure1_MDS_Plots\MDS_Plots.m"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "mds_labeled_longrun_vs_60epoch_outputs"
GROUP_NAMES = ("medium", "good", "great")
RUN_LABELS = ("Data", "60 epoch", "LongRun")
RUN_COLORS = ("#111827", "#2563eb", "#16a34a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LongRun vs 60 epoch for medium/good/great cells."
    )
    parser.add_argument("--baseline-mat", type=Path, default=DEFAULT_BASELINE_MAT)
    parser.add_argument("--longrun-mat", type=Path, default=DEFAULT_LONGRUN_MAT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mds-script", type=Path, default=DEFAULT_MDS_SCRIPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_mds_cell_groups(mds_script: Path) -> dict[str, list[int]]:
    text = mds_script.read_text(encoding="utf-8", errors="replace")
    groups: dict[str, list[int]] = {}

    for group in GROUP_NAMES:
        pattern = rf"(?m)^\s*{group}\s*=\s*\[(.*?)\]\s*;"
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not find a '{group}' cell list in {mds_script}")
        groups[group] = [int(value) for value in re.findall(r"\d+", match.group(1))]

    seen: set[int] = set()
    duplicates: list[int] = []
    for cells in groups.values():
        for cell in cells:
            if cell in seen:
                duplicates.append(cell)
            seen.add(cell)
    if duplicates:
        raise ValueError(f"Duplicate medium/good/great cell IDs: {duplicates}")

    return groups


def ordered_labeled_cells(groups: dict[str, list[int]]) -> tuple[list[int], dict[int, str]]:
    cell_ids: list[int] = []
    group_by_cell: dict[int, str] = {}
    for group in GROUP_NAMES:
        cells = groups[group]
        cell_ids.extend(cells)
        group_by_cell.update({cell: group for cell in cells})
    return cell_ids, group_by_cell


def subset_indices(all_cell_ids: list[int], selected_cell_ids: list[int]) -> np.ndarray:
    position_by_cell = {cell_id: idx for idx, cell_id in enumerate(all_cell_ids)}
    missing = [cell_id for cell_id in selected_cell_ids if cell_id not in position_by_cell]
    if missing:
        raise ValueError(f"Selected cell IDs are not in simulation_config cell_targets: {missing}")
    return np.asarray([position_by_cell[cell_id] for cell_id in selected_cell_ids])


def subset_run_metrics(run: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    fields = [
        "best_batches",
        "best_losses",
        "cvs",
        "firing_rates",
        "selected_spikes",
    ]
    return {field: run[field][indices] for field in fields}


def metric_summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.nanmean(values)), float(np.nanmedian(values))


def summarize_group(
    group_name: str,
    group_mask: np.ndarray,
    baseline: dict[str, np.ndarray],
    longrun: dict[str, np.ndarray],
    data_cvs: np.ndarray,
    data_rates: np.ndarray,
) -> dict[str, float | int | str]:
    delta = longrun["best_losses"][group_mask] - baseline["best_losses"][group_mask]
    baseline_losses = baseline["best_losses"][group_mask]
    longrun_losses = longrun["best_losses"][group_mask]
    baseline_cvs = baseline["cvs"][group_mask]
    longrun_cvs = longrun["cvs"][group_mask]
    baseline_rates = baseline["firing_rates"][group_mask]
    longrun_rates = longrun["firing_rates"][group_mask]
    group_data_cvs = data_cvs[group_mask]
    group_data_rates = data_rates[group_mask]

    mean_60_loss, median_60_loss = metric_summary(baseline_losses)
    mean_long_loss, median_long_loss = metric_summary(longrun_losses)
    mean_data_cv, median_data_cv = metric_summary(group_data_cvs)
    mean_60_cv, median_60_cv = metric_summary(baseline_cvs)
    mean_long_cv, median_long_cv = metric_summary(longrun_cvs)
    mean_data_rate, median_data_rate = metric_summary(group_data_rates)
    mean_60_rate, median_60_rate = metric_summary(baseline_rates)
    mean_long_rate, median_long_rate = metric_summary(longrun_rates)

    return {
        "group": group_name,
        "cells": int(np.sum(group_mask)),
        "longrun_lower_loss_cells": int(np.sum(delta < 0)),
        "longrun_higher_loss_cells": int(np.sum(delta > 0)),
        "equal_loss_cells": int(np.sum(delta == 0)),
        "mean_best_psth_sse_60epoch": mean_60_loss,
        "mean_best_psth_sse_longrun": mean_long_loss,
        "median_best_psth_sse_60epoch": median_60_loss,
        "median_best_psth_sse_longrun": median_long_loss,
        "mean_loss_delta_longrun_minus_60epoch": float(np.nanmean(delta)),
        "median_loss_delta_longrun_minus_60epoch": float(np.nanmedian(delta)),
        "mean_cv_data": mean_data_cv,
        "mean_cv_60epoch": mean_60_cv,
        "mean_cv_longrun": mean_long_cv,
        "median_cv_data": median_data_cv,
        "median_cv_60epoch": median_60_cv,
        "median_cv_longrun": median_long_cv,
        "mean_fr_hz_data": mean_data_rate,
        "mean_fr_hz_60epoch": mean_60_rate,
        "mean_fr_hz_longrun": mean_long_rate,
        "median_fr_hz_data": median_data_rate,
        "median_fr_hz_60epoch": median_60_rate,
        "median_fr_hz_longrun": median_long_rate,
    }


def write_tables(
    out_dir: Path,
    selected_cell_ids: list[int],
    group_by_cell: dict[int, str],
    baseline: dict[str, np.ndarray],
    longrun: dict[str, np.ndarray],
    data_cvs: np.ndarray,
    data_rates: np.ndarray,
) -> list[dict[str, float | int | str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    delta = longrun["best_losses"] - baseline["best_losses"]

    per_cell_path = out_dir / "mds_labeled_comparison_metrics_by_cell.csv"
    with per_cell_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cell_id",
                "quality_group",
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
        for idx, cell_id in enumerate(selected_cell_ids):
            writer.writerow(
                [
                    cell_id,
                    group_by_cell[cell_id],
                    int(baseline["best_batches"][idx]),
                    int(baseline["best_batches"][idx]) + 1,
                    float(baseline["best_losses"][idx]),
                    int(longrun["best_batches"][idx]),
                    int(longrun["best_batches"][idx]) + 1,
                    float(longrun["best_losses"][idx]),
                    float(delta[idx]),
                    bool(delta[idx] < 0),
                    float(data_cvs[idx]),
                    float(baseline["cvs"][idx]),
                    float(longrun["cvs"][idx]),
                    float(data_rates[idx]),
                    float(baseline["firing_rates"][idx]),
                    float(longrun["firing_rates"][idx]),
                    int(baseline["selected_spikes"][idx]),
                    int(longrun["selected_spikes"][idx]),
                ]
            )

    group_names = list(GROUP_NAMES) + ["medium+good+great"]
    summaries = []
    cell_groups = np.asarray([group_by_cell[cell_id] for cell_id in selected_cell_ids])
    for group in group_names:
        if group == "medium+good+great":
            mask = np.ones(len(selected_cell_ids), dtype=bool)
        else:
            mask = cell_groups == group
        summaries.append(
            summarize_group(group, mask, baseline, longrun, data_cvs, data_rates)
        )

    summary_csv_path = out_dir / "mds_labeled_summary_by_group.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summaries[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    with (out_dir / "mds_labeled_summary_by_group.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summaries, handle, indent=2)

    print(f"Wrote table: {per_cell_path}")
    print(f"Wrote group summary: {summary_csv_path}")
    return summaries


def group_ranges(selected_cell_ids: list[int], group_by_cell: dict[int, str]) -> list[tuple[str, int, int]]:
    ranges = []
    start = 0
    current_group = group_by_cell[selected_cell_ids[0]]
    for idx, cell_id in enumerate(selected_cell_ids):
        group = group_by_cell[cell_id]
        if group != current_group:
            ranges.append((current_group, start, idx - 1))
            start = idx
            current_group = group
    ranges.append((current_group, start, len(selected_cell_ids) - 1))
    return ranges


def decorate_grouped_axis(
    ax: plt.Axes, selected_cell_ids: list[int], group_by_cell: dict[int, str]
) -> None:
    ranges = group_ranges(selected_cell_ids, group_by_cell)
    for group, start, stop in ranges:
        center = (start + stop) / 2
        ax.text(
            center,
            0.985,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
        if stop < len(selected_cell_ids) - 1:
            ax.axvline(stop + 0.5, color="#9ca3af", lw=0.8, ls="--", alpha=0.8)


def selected_tick_labels(selected_cell_ids: list[int]) -> list[str]:
    return [str(cell_id) for cell_id in selected_cell_ids]


def plot_labeled_loss(
    out_dir: Path,
    selected_cell_ids: list[int],
    group_by_cell: dict[int, str],
    baseline: dict[str, np.ndarray],
    longrun: dict[str, np.ndarray],
    summaries: list[dict[str, float | int | str]],
    dpi: int,
) -> plt.Figure:
    x = np.arange(len(selected_cell_ids))
    delta = longrun["best_losses"] - baseline["best_losses"]
    colors = np.where(delta < 0, "#15803d", np.where(delta > 0, "#b91c1c", "#6b7280"))

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    axes[0].bar(x, delta, color=colors, width=0.82)
    axes[0].axhline(0, color="#111827", lw=0.8)
    axes[0].set_title("Best PSTH SSE delta for MDS-labeled cells (LongRun - 60 epoch)")
    axes[0].set_ylabel("SSE delta")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(selected_tick_labels(selected_cell_ids), rotation=90, fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)
    decorate_grouped_axis(axes[0], selected_cell_ids, group_by_cell)

    group_summaries = summaries[: len(GROUP_NAMES)]
    group_labels = [str(summary["group"]) for summary in group_summaries]
    mean_60 = [float(summary["mean_best_psth_sse_60epoch"]) for summary in group_summaries]
    mean_long = [float(summary["mean_best_psth_sse_longrun"]) for summary in group_summaries]
    gx = np.arange(len(group_labels))
    axes[1].bar(gx - 0.18, mean_60, width=0.36, color="#2563eb", label="60 epoch")
    axes[1].bar(gx + 0.18, mean_long, width=0.36, color="#16a34a", label="LongRun")
    axes[1].set_xticks(gx)
    axes[1].set_xticklabels(group_labels)
    axes[1].set_ylabel("Mean best PSTH SSE")
    axes[1].set_title("Mean best PSTH SSE by quality group")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(out_dir / "mds_labeled_loss_comparison.png", dpi=dpi)
    return fig


def plot_grouped_metric(
    out_dir: Path,
    selected_cell_ids: list[int],
    group_by_cell: dict[int, str],
    data_values: np.ndarray,
    baseline_values: np.ndarray,
    longrun_values: np.ndarray,
    summaries: list[dict[str, float | int | str]],
    metric_name: str,
    metric_ylabel: str,
    summary_keys: tuple[str, str, str],
    out_name: str,
    dpi: int,
) -> plt.Figure:
    x = np.arange(len(selected_cell_ids))
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)

    axes[0].plot(x, data_values, color=RUN_COLORS[0], lw=1.2, marker=".", label="Data")
    axes[0].plot(x, baseline_values, color=RUN_COLORS[1], lw=1.2, marker=".", label="60 epoch")
    axes[0].plot(x, longrun_values, color=RUN_COLORS[2], lw=1.2, marker=".", label="LongRun")
    axes[0].set_title(f"{metric_name} for MDS-labeled cells")
    axes[0].set_ylabel(metric_ylabel)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(selected_tick_labels(selected_cell_ids), rotation=90, fontsize=7)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    decorate_grouped_axis(axes[0], selected_cell_ids, group_by_cell)

    group_summaries = summaries[: len(GROUP_NAMES)]
    group_labels = [str(summary["group"]) for summary in group_summaries]
    gx = np.arange(len(group_labels))
    width = 0.24
    for offset, key, label, color in zip(
        (-width, 0.0, width), summary_keys, RUN_LABELS, RUN_COLORS
    ):
        means = [float(summary[key]) for summary in group_summaries]
        axes[1].bar(gx + offset, means, width=width, color=color, label=label)
    axes[1].set_xticks(gx)
    axes[1].set_xticklabels(group_labels)
    axes[1].set_ylabel(metric_ylabel)
    axes[1].set_title(f"Mean {metric_name.lower()} by quality group")
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(out_dir / out_name, dpi=dpi)
    return fig


def write_plots(
    out_dir: Path,
    selected_cell_ids: list[int],
    group_by_cell: dict[int, str],
    baseline: dict[str, np.ndarray],
    longrun: dict[str, np.ndarray],
    data_cvs: np.ndarray,
    data_rates: np.ndarray,
    summaries: list[dict[str, float | int | str]],
    dpi: int,
) -> None:
    figures = [
        plot_labeled_loss(
            out_dir,
            selected_cell_ids,
            group_by_cell,
            baseline,
            longrun,
            summaries,
            dpi,
        ),
        plot_grouped_metric(
            out_dir,
            selected_cell_ids,
            group_by_cell,
            data_cvs,
            baseline["cvs"],
            longrun["cvs"],
            summaries,
            "CV",
            "ISI CV",
            ("mean_cv_data", "mean_cv_60epoch", "mean_cv_longrun"),
            "mds_labeled_cv_comparison.png",
            dpi,
        ),
        plot_grouped_metric(
            out_dir,
            selected_cell_ids,
            group_by_cell,
            data_rates,
            baseline["firing_rates"],
            longrun["firing_rates"],
            summaries,
            "Firing rate",
            "Hz",
            ("mean_fr_hz_data", "mean_fr_hz_60epoch", "mean_fr_hz_longrun"),
            "mds_labeled_firing_rate_comparison.png",
            dpi,
        ),
    ]

    pdf_path = out_dir / "mds_labeled_summary_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote plots PDF: {pdf_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = parse_mds_cell_groups(args.mds_script)
    selected_cell_ids, group_by_cell = ordered_labeled_cells(groups)
    print(
        "Loaded MDS labels: "
        + ", ".join(f"{group}={len(cells)}" for group, cells in groups.items())
        + f"; total={len(selected_cell_ids)}"
    )

    config = load_config(args.config)
    simulation = config["simulation"]
    all_cell_ids = [int(cell) for cell in simulation["cell_targets"]]
    indices = subset_indices(all_cell_ids, selected_cell_ids)

    dt_ms = float(simulation["dt"])
    bin_steps = int(simulation["PSTH_granularity"])
    expected_batches = int(simulation["batch_size"])
    expected_trials = 10
    expected_cells = len(all_cell_ids)
    expected_steps = int(simulation["sim_len"])

    print("Building data raster")
    data_raster = load_data_raster(config, all_cell_ids, expected_steps, dt_ms)
    data_psth = psth_from_data_raster(data_raster, bin_steps)
    all_data_cvs, all_data_rates = data_metrics(data_raster, dt_ms)
    data_cvs = all_data_cvs[indices]
    data_rates = all_data_rates[indices]

    baseline_all = process_run(
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
    baseline = subset_run_metrics(baseline_all, indices)
    del baseline_all
    gc.collect()

    longrun_all = process_run(
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
    longrun = subset_run_metrics(longrun_all, indices)
    del longrun_all
    gc.collect()

    summaries = write_tables(
        args.out_dir,
        selected_cell_ids,
        group_by_cell,
        baseline,
        longrun,
        data_cvs,
        data_rates,
    )
    write_plots(
        args.out_dir,
        selected_cell_ids,
        group_by_cell,
        baseline,
        longrun,
        data_cvs,
        data_rates,
        summaries,
        args.dpi,
    )

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
