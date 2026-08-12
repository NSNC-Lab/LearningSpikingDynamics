"""Plot best-batch raster/PSTH comparisons for 60_Epoch_allCells_withRef.mat.

The script computes PSTH SSE for every batch/cell pair, chooses the best batch
per cell, then writes a 22-page PDF plus one PNG per page.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT_MAT = REPO_ROOT / "80_epoch_checkpoint_strffix4.mat"
DEFAULT_CONFIG = REPO_ROOT / "simulation_config.yaml"
DEFAULT_OUT_DIR = SCRIPT_DIR / "outputs2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create best-batch raster/PSTH comparison pages."
    )
    parser.add_argument("--mat", type=Path, default=DEFAULT_OUTPUT_MAT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cells-per-page", type=int, default=10)
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


def load_data_raster(config: dict, cell_ids: list[int], sim_len: int, dt_ms: float) -> np.ndarray:
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
            valid = trial_times[(trial_times >= 0) & (trial_times < sim_len / sample_rate)]
            spike_indices = (valid * sample_rate).astype(np.int64)
            spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < sim_len)]
            raster[cell_pos, trial, spike_indices] = 1

    return raster


def psth_from_sim_output(output: np.ndarray, bin_steps: int) -> np.ndarray:
    batches, trials, cells, steps = output.shape
    bins = steps // bin_steps
    trimmed = output[:, :, :, : bins * bin_steps]
    binned = trimmed.reshape(batches, trials, cells, bins, bin_steps)
    return binned.sum(axis=(1, 4)).astype(np.float32)


def psth_from_data_raster(raster: np.ndarray, bin_steps: int) -> np.ndarray:
    cells, trials, steps = raster.shape
    bins = steps // bin_steps
    trimmed = raster[:, :, : bins * bin_steps]
    binned = trimmed.reshape(cells, trials, bins, bin_steps)
    return binned.sum(axis=(1, 3)).astype(np.float32)


def compute_best_batches(
    output: np.ndarray, data_raster: np.ndarray, bin_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sim_psth = psth_from_sim_output(output, bin_steps)
    data_psth = psth_from_data_raster(data_raster, bin_steps)
    psth_errors = sim_psth - data_psth[None, :, :]
    losses = np.sum(psth_errors**2, axis=2)
    best_batches = np.argmin(losses, axis=0)
    best_losses = losses[best_batches, np.arange(losses.shape[1])]
    return best_batches, best_losses, losses, sim_psth


def save_loss_tables(
    out_dir: Path,
    cell_ids: list[int],
    best_batches: np.ndarray,
    best_losses: np.ndarray,
    losses: np.ndarray,
    output: np.ndarray,
    data_raster: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "psth_losses_by_batch_cell.npy", losses)

    with (out_dir / "best_psth_losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cell_id",
                "best_batch_zero_based",
                "best_batch_one_based",
                "best_psth_sse",
                "best_sim_spikes",
                "data_spikes",
            ]
        )
        for cell_pos, cell_id in enumerate(cell_ids):
            batch = int(best_batches[cell_pos])
            writer.writerow(
                [
                    cell_id,
                    batch,
                    batch + 1,
                    float(best_losses[cell_pos]),
                    int(output[batch, :, cell_pos, :].sum()),
                    int(data_raster[cell_pos, :, :].sum()),
                ]
            )


def plot_raster(ax, raster: np.ndarray, dt_ms: float, color: str) -> None:
    trials, _ = raster.shape
    trial_idx, time_idx = np.nonzero(raster)
    ax.scatter(
        time_idx * dt_ms / 1000.0,
        trial_idx + 1,
        s=16,
        marker="|",
        linewidths=0.9,
        color=color,
    )
    ax.set_ylim(0.5, trials + 0.5)
    ax.set_yticks([1, trials])
    ax.set_xlim(0, raster.shape[1] * dt_ms / 1000.0)
    ax.tick_params(axis="both", labelsize=7, length=3, width=0.8)


def plot_pages(
    out_dir: Path,
    cell_ids: list[int],
    output: np.ndarray,
    data_raster: np.ndarray,
    sim_psth: np.ndarray,
    data_psth: np.ndarray,
    best_batches: np.ndarray,
    best_losses: np.ndarray,
    dt_ms: float,
    bin_steps: int,
    cells_per_page: int,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_dir = out_dir / "page_pngs"
    page_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "best_raster_psth_comparison_pages.pdf"

    bins = data_psth.shape[1]
    bin_centers = (np.arange(bins) + 0.5) * bin_steps * dt_ms / 1000.0
    total_pages = int(np.ceil(len(cell_ids) / cells_per_page))

    with PdfPages(pdf_path) as pdf:
        for page in range(total_pages):
            start = page * cells_per_page
            stop = min(start + cells_per_page, len(cell_ids))
            page_cells = list(range(start, stop))
            rows = len(page_cells)

            fig, axes = plt.subplots(
                rows,
                3,
                figsize=(15, max(1.35 * rows, 3.0)),
                squeeze=False,
                gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]},
            )
            fig.suptitle(
                f"Best batch raster/PSTH comparison - page {page + 1} of {total_pages}",
                fontsize=14,
                y=0.995,
            )

            axes[0, 0].set_title("Best simulation raster", fontsize=10)
            axes[0, 1].set_title("Data raster", fontsize=10)
            axes[0, 2].set_title("PSTH overlay", fontsize=10)

            for row, cell_pos in enumerate(page_cells):
                cell_id = cell_ids[cell_pos]
                batch = int(best_batches[cell_pos])
                sim_raster = output[batch, :, cell_pos, :]
                data_cell_raster = data_raster[cell_pos, :, :]

                ax_sim, ax_data, ax_psth = axes[row]
                plot_raster(ax_sim, sim_raster, dt_ms, "#2563eb")
                plot_raster(ax_data, data_cell_raster, dt_ms, "#111827")

                ax_sim.set_ylabel(f"Cell {cell_id}\nB{batch + 1}\nSSE {best_losses[cell_pos]:.0f}", fontsize=7)
                if row == rows - 1:
                    ax_sim.set_xlabel("Time (s)", fontsize=8)
                    ax_data.set_xlabel("Time (s)", fontsize=8)
                else:
                    ax_sim.set_xticklabels([])
                    ax_data.set_xticklabels([])

                ax_psth.plot(bin_centers, data_psth[cell_pos], color="#111827", lw=1.2, label="Data")
                ax_psth.plot(bin_centers, sim_psth[batch, cell_pos], color="#2563eb", lw=1.0, label="Simulation")
                ax_psth.set_xlim(0, data_cell_raster.shape[1] * dt_ms / 1000.0)
                ax_psth.tick_params(axis="both", labelsize=7, length=3, width=0.8)
                if row == rows - 1:
                    ax_psth.set_xlabel("Time (s)", fontsize=8)
                else:
                    ax_psth.set_xticklabels([])
                ax_psth.set_ylabel("Spikes/bin", fontsize=7)
                if row == 0:
                    ax_psth.legend(loc="upper right", fontsize=7, frameon=False)

            fig.tight_layout(rect=[0, 0, 1, 0.982], h_pad=0.28, w_pad=0.45)
            png_path = page_dir / f"page_{page + 1:02d}.png"
            fig.savefig(png_path, dpi=dpi)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Wrote page {page + 1:02d}/{total_pages}: {png_path}")

    print(f"Wrote PDF: {pdf_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    simulation = config["simulation"]
    cell_ids = [int(cell) for cell in simulation["cell_targets"]]
    dt_ms = float(simulation["dt"])
    bin_steps = int(simulation["PSTH_granularity"])

    mat = loadmat(args.mat, squeeze_me=False, struct_as_record=False)
    output = np.asarray(mat["output"], dtype=np.uint8)
    if output.ndim != 4:
        raise ValueError(f"Expected output to have 4 dimensions, got shape {output.shape}")

    sim_len = output.shape[-1]
    if len(cell_ids) != output.shape[2]:
        raise ValueError(
            f"Config has {len(cell_ids)} cell targets, but output has {output.shape[2]} cells."
        )

    print(f"Loaded output: batches={output.shape[0]}, trials={output.shape[1]}, cells={output.shape[2]}, steps={sim_len}")
    data_raster = load_data_raster(config, cell_ids, sim_len, dt_ms)
    data_psth = psth_from_data_raster(data_raster, bin_steps)
    best_batches, best_losses, losses, sim_psth = compute_best_batches(
        output, data_raster, bin_steps
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_loss_tables(
        args.out_dir,
        cell_ids,
        best_batches,
        best_losses,
        losses,
        output,
        data_raster,
    )
    plot_pages(
        args.out_dir,
        cell_ids,
        output,
        data_raster,
        sim_psth,
        data_psth,
        best_batches,
        best_losses,
        dt_ms,
        bin_steps,
        args.cells_per_page,
        args.dpi,
    )


if __name__ == "__main__":
    main()
