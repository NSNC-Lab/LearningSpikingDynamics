from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAT = REPO_ROOT / "100_epoch_post_strf_fix_4.mat"
DEFAULT_OUT_DIR = REPO_ROOT / "Archive" / "Debugging" / "60_epoch_all_cell_post_strf_fix_3"

PARAMETER_ORDER = [
    "strf_gain",
    "strf_alpha",
    "output_ad",
    "abs_ref",
    "rel_ref_a",
    "rel_ref_b",
    "rel_ref_c",
    "on_ron_gsyn",
    "off_ron_gsyn",
    "on_sonoff_gsyn",
    "off_sonoff_gsyn",
    "sonoff_ron_gsyn",
]

# Older compressed E-prop outputs store a numeric tracker with axes
# epoch x parameter x cell x batch instead of a MATLAB struct of named fields.
LEGACY_PARAMETER_ORDER = [
    "strf_gain",
    "strf_alpha",
    "output_ad",
    "on_ron_gsyn",
    "off_ron_gsyn",
    "on_sonoff_gsyn",
    "off_sonoff_gsyn",
    "sonoff_ron_gsyn",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot population-level parameter convergence from saved batch x cell x epoch trackers."
    )
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cell-alpha", type=float, default=0.10)
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=0,
        help="Cumulative epoch represented by the first saved parameter state (default: 0).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = load_params(args.mat)
    fields = ordered_param_names(params)

    pdf_path = args.out_dir / "population_parameter_convergence_index.pdf"
    csv_path = args.out_dir / "population_parameter_convergence_summary.csv"
    learning_pdf_path = args.out_dir / "population_parameter_learning_rate.pdf"
    learning_csv_path = args.out_dir / "population_parameter_learning_rate_summary.csv"
    plot_convergence_pdf(pdf_path, params, fields, args.cell_alpha, args.epoch_start)
    write_summary_csv(csv_path, params, fields)
    plot_learning_rate_pdf(learning_pdf_path, params, fields, args.cell_alpha, args.epoch_start)
    write_learning_rate_summary_csv(learning_csv_path, params, fields)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {learning_pdf_path}")
    print(f"Wrote {learning_csv_path}")


def load_params(mat_path: Path):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "params" not in mat:
        raise KeyError(f"{mat_path} does not contain a 'params' variable.")
    params = mat["params"]
    if getattr(params, "_fieldnames", None):
        return params

    arr = np.asarray(params, dtype=float)
    if arr.ndim == 3 and arr.shape[0] == len(LEGACY_PARAMETER_ORDER):
        # A singleton epoch axis was removed by squeeze_me=True.
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[1] != len(LEGACY_PARAMETER_ORDER):
        raise ValueError(
            "Numeric 'params' must have legacy shape "
            f"epoch x {len(LEGACY_PARAMETER_ORDER)} parameters x cell x batch; got {arr.shape}."
        )

    trackers = {
        name: np.transpose(arr[:, index, :, :], (2, 1, 0))
        for index, name in enumerate(LEGACY_PARAMETER_ORDER)
    }
    # Legacy files store STRF alpha in seconds; current trackers use milliseconds.
    # The normalized metrics are scale-invariant, but this keeps raw movement
    # summaries comparable between legacy and current runs.
    trackers["strf_alpha"] = trackers["strf_alpha"] * 1000.0
    return trackers


def ordered_param_names(params) -> list[str]:
    if isinstance(params, dict):
        fields = list(params)
    else:
        fields = list(getattr(params, "_fieldnames", []))
    ordered = [name for name in PARAMETER_ORDER if name in fields]
    ordered.extend(name for name in fields if name not in ordered)
    return ordered


def as_batch_cell_epoch(params, name: str) -> np.ndarray:
    value = params[name] if isinstance(params, dict) else getattr(params, name)
    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected batch x cell x epoch, got shape {arr.shape}.")
    return arr


def convergence_index(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread = np.nanstd(arr, axis=0)
    initial = spread[:, :1]
    valid = initial > 1e-12
    remaining_spread = np.full_like(spread, np.nan, dtype=float)
    np.divide(spread, initial, out=remaining_spread, where=valid)
    convergence = 1.0 - remaining_spread
    return convergence, remaining_spread


def movement_ratio(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    movement = np.nanmean(np.abs(np.diff(arr, axis=2)), axis=0)
    baseline = movement[:, :1]
    valid = baseline > 1e-12
    ratio = np.full_like(movement, np.nan, dtype=float)
    np.divide(movement, baseline, out=ratio, where=valid)
    learned_index = 1.0 - ratio
    return movement, ratio, learned_index


def plot_convergence_pdf(
    pdf_path: Path, params, fields: list[str], cell_alpha: float, epoch_start: int
) -> None:
    with PdfPages(pdf_path) as pdf:
        for name in fields:
            arr = as_batch_cell_epoch(params, name)
            conv, remaining = convergence_index(arr)
            epochs = np.arange(epoch_start, epoch_start + conv.shape[1])
            mean_conv = np.nanmean(conv, axis=0)
            median_conv = np.nanmedian(conv, axis=0)
            final_conv = conv[:, -1]

            fig, ax = plt.subplots(figsize=(11, 7.5))
            ax.plot(epochs, conv.T, color="#4c78a8", alpha=cell_alpha, linewidth=0.65)
            ax.plot(epochs, mean_conv, color="#0b3c5d", linewidth=3.2, label="population mean")
            ax.plot(epochs, median_conv, color="#bf5700", linewidth=2.0, linestyle="--", label="population median")
            ax.axhline(0, color="black", linewidth=0.9, alpha=0.55, label="unchanged spread")
            ax.axhline(1, color="forestgreen", linewidth=0.9, alpha=0.40, linestyle=":", label="complete collapse")

            lower, upper = robust_ylim(conv)
            ax.set_ylim(lower, upper)
            ax.set_xlim(epochs[0], max(epochs[0] + 1, epochs[-1]))
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"Convergence index: 1 - batch std / epoch-{epoch_start} batch std")
            ax.set_title(f"Population convergence by cell: {name}")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")

            text = summary_text(name, final_conv, remaining[:, -1])
            ax.text(
                0.015,
                0.025,
                text,
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_learning_rate_pdf(
    pdf_path: Path, params, fields: list[str], cell_alpha: float, epoch_start: int
) -> None:
    with PdfPages(pdf_path) as pdf:
        for name in fields:
            arr = as_batch_cell_epoch(params, name)
            movement, ratio, learned = movement_ratio(arr)
            epochs = np.arange(epoch_start + 1, epoch_start + 1 + ratio.shape[1])
            mean_ratio = np.nanmean(ratio, axis=0)
            median_ratio = np.nanmedian(ratio, axis=0)
            final_ratio = ratio[:, -1]
            final_movement = movement[:, -1]

            fig, ax = plt.subplots(figsize=(11, 7.5))
            ax.plot(epochs, ratio.T, color="#6f92b7", alpha=cell_alpha, linewidth=0.65)
            ax.plot(epochs, mean_ratio, color="#0b3c5d", linewidth=3.2, label="population mean")
            ax.plot(epochs, median_ratio, color="#bf5700", linewidth=2.0, linestyle="--", label="population median")
            ax.axhline(
                1,
                color="black",
                linewidth=0.9,
                alpha=0.55,
                label=f"epoch-{epoch_start + 1} movement",
            )
            ax.axhline(0, color="forestgreen", linewidth=0.9, alpha=0.40, linestyle=":", label="no movement")

            lower, upper = robust_ratio_ylim(ratio)
            ax.set_ylim(lower, upper)
            ax.set_xlim(epochs[0], max(epochs[0] + 1, epochs[-1]))
            ax.set_xlabel("Epoch transition ending at epoch")
            ax.set_ylabel(
                "Movement ratio: mean batch |delta parameter| / "
                f"epoch-{epoch_start + 1} movement"
            )
            ax.set_title(f"Population learning rate by cell: {name}")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")

            text = learning_summary_text(name, final_ratio, final_movement, learned[:, -1])
            ax.text(
                0.015,
                0.025,
                text,
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def robust_ylim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lower = min(-0.25, float(np.nanpercentile(finite, 1)))
    upper = max(1.05, float(np.nanpercentile(finite, 99)))
    pad = 0.08 * max(1e-6, upper - lower)
    return lower - pad, upper + pad


def robust_ratio_ylim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.25
    lower = 0.0
    upper = max(1.25, float(np.nanpercentile(finite, 99)))
    pad = 0.08 * max(1e-6, upper - lower)
    return lower, upper + pad


def summary_text(name: str, final_conv: np.ndarray, final_remaining: np.ndarray) -> str:
    improved = np.nanmean(final_conv > 0) * 100
    half_spread = np.nanmean(final_remaining <= 0.5) * 100
    spread_out = np.nanmean(final_remaining > 1.0) * 100
    return (
        f"Final {name}\n"
        f"mean convergence: {np.nanmean(final_conv):.3g}\n"
        f"median convergence: {np.nanmedian(final_conv):.3g}\n"
        f"cells with lower spread: {improved:.1f}%\n"
        f"cells with <=50% initial spread: {half_spread:.1f}%\n"
        f"cells with increased spread: {spread_out:.1f}%"
    )


def learning_summary_text(name: str, final_ratio: np.ndarray, final_movement: np.ndarray, final_learned: np.ndarray) -> str:
    tenth = np.nanmean(final_ratio <= 0.1) * 100
    quarter = np.nanmean(final_ratio <= 0.25) * 100
    still_moving = np.nanmean(final_ratio >= 1.0) * 100
    return (
        f"Final {name}\n"
        f"mean movement ratio: {np.nanmean(final_ratio):.3g}\n"
        f"median movement ratio: {np.nanmedian(final_ratio):.3g}\n"
        f"mean learned index: {np.nanmean(final_learned):.3g}\n"
        f"cells <=10% initial movement: {tenth:.1f}%\n"
        f"cells <=25% initial movement: {quarter:.1f}%\n"
        f"cells still moving >=initial: {still_moving:.1f}%\n"
        f"mean raw final movement: {np.nanmean(final_movement):.3g}"
    )


def write_summary_csv(csv_path: Path, params, fields: list[str]) -> None:
    columns = [
        "parameter",
        "final_mean_convergence",
        "final_median_convergence",
        "final_mean_remaining_spread",
        "final_median_remaining_spread",
        "percent_cells_lower_spread",
        "percent_cells_half_or_less_initial_spread",
        "percent_cells_increased_spread",
        "n_cells",
        "n_epochs",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name in fields:
            arr = as_batch_cell_epoch(params, name)
            conv, remaining = convergence_index(arr)
            final_conv = conv[:, -1]
            final_remaining = remaining[:, -1]
            writer.writerow(
                {
                    "parameter": name,
                    "final_mean_convergence": np.nanmean(final_conv),
                    "final_median_convergence": np.nanmedian(final_conv),
                    "final_mean_remaining_spread": np.nanmean(final_remaining),
                    "final_median_remaining_spread": np.nanmedian(final_remaining),
                    "percent_cells_lower_spread": np.nanmean(final_conv > 0) * 100,
                    "percent_cells_half_or_less_initial_spread": np.nanmean(final_remaining <= 0.5) * 100,
                    "percent_cells_increased_spread": np.nanmean(final_remaining > 1.0) * 100,
                    "n_cells": conv.shape[0],
                    "n_epochs": conv.shape[1],
                }
            )


def write_learning_rate_summary_csv(csv_path: Path, params, fields: list[str]) -> None:
    columns = [
        "parameter",
        "final_mean_movement_ratio",
        "final_median_movement_ratio",
        "final_mean_learned_index",
        "final_median_learned_index",
        "final_mean_raw_movement",
        "final_median_raw_movement",
        "percent_cells_10pct_or_less_initial_movement",
        "percent_cells_25pct_or_less_initial_movement",
        "percent_cells_still_moving_at_or_above_initial",
        "n_cells",
        "n_epoch_transitions",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name in fields:
            arr = as_batch_cell_epoch(params, name)
            movement, ratio, learned = movement_ratio(arr)
            final_ratio = ratio[:, -1]
            final_learned = learned[:, -1]
            final_movement = movement[:, -1]
            writer.writerow(
                {
                    "parameter": name,
                    "final_mean_movement_ratio": np.nanmean(final_ratio),
                    "final_median_movement_ratio": np.nanmedian(final_ratio),
                    "final_mean_learned_index": np.nanmean(final_learned),
                    "final_median_learned_index": np.nanmedian(final_learned),
                    "final_mean_raw_movement": np.nanmean(final_movement),
                    "final_median_raw_movement": np.nanmedian(final_movement),
                    "percent_cells_10pct_or_less_initial_movement": np.nanmean(final_ratio <= 0.1) * 100,
                    "percent_cells_25pct_or_less_initial_movement": np.nanmean(final_ratio <= 0.25) * 100,
                    "percent_cells_still_moving_at_or_above_initial": np.nanmean(final_ratio >= 1.0) * 100,
                    "n_cells": ratio.shape[0],
                    "n_epoch_transitions": ratio.shape[1],
                }
            )


if __name__ == "__main__":
    main()
