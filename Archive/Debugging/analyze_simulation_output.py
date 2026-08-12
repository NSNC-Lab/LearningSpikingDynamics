from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import yaml
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "60_epoch_all_cell_post_strf_fix_3.mat"
DEFAULT_CONFIG = REPO_ROOT / "simulation_config.yaml"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "Archive" / "Debugging" / "simulation_output_analysis"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze simulation_output.mat parameter histories and raster fits.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--max-raster-pages", type=int, default=60)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    mat = sio.loadmat(args.output, squeeze_me=False, struct_as_record=False)
    output = normalize_output(mat["output"])
    losses = np.ravel(mat.get("losses", np.array([], dtype=float))).astype(float)
    n_batch, n_trial, n_cell, sim_len = output.shape

    config["sim_len"] = sim_len
    cell_targets = normalize_cell_targets(config, n_cell)
    data_raster = build_data_raster(config, cell_targets, n_trial)
    data_for_psth = data_raster.transpose(1, 0, 2)[None, :, :, :]
    data_psth, bin_centers = raster_to_psth(data_for_psth, config["psth_granularity"], config["dt_ms"])
    data_psth = data_psth[0]
    sim_psth, _ = raster_to_psth(output, config["psth_granularity"], config["dt_ms"])

    psth_loss = np.mean((sim_psth - data_psth[None, :, :]) ** 2, axis=2)
    data_cv = np.array([cv_from_raster(data_raster[cell_index]) for cell_index in range(n_cell)])
    sim_cv = np.zeros((n_batch, n_cell), dtype=float)
    cv_loss = np.zeros((n_batch, n_cell), dtype=float)
    for batch_index in range(n_batch):
        for cell_index in range(n_cell):
            sim_cv[batch_index, cell_index] = cv_from_raster(output[batch_index, :, cell_index, :])
            cv_loss[batch_index, cell_index] = (sim_cv[batch_index, cell_index] - data_cv[cell_index]) ** 2

    cv_loss_sortable = np.nan_to_num(cv_loss, nan=np.inf, posinf=np.inf, neginf=np.inf)
    psth_loss_sortable = np.nan_to_num(psth_loss, nan=np.inf, posinf=np.inf, neginf=np.inf)
    combined_rank = ranks(psth_loss_sortable.ravel()).reshape(psth_loss.shape) + ranks(cv_loss_sortable.ravel()).reshape(cv_loss.shape)

    best = {
        "psth": tuple(np.unravel_index(np.argmin(psth_loss_sortable), psth_loss.shape)),
        "cv": tuple(np.unravel_index(np.argmin(cv_loss_sortable), cv_loss.shape)),
        "combined": tuple(np.unravel_index(np.argmin(combined_rank), combined_rank.shape)),
    }

    params = get_params(args.output)
    param_histories = {
        name: normalize_param_history(getattr(params, name), n_batch, n_cell, len(losses))
        for name in ordered_param_names(params)
    }

    parameter_pdf = args.out_dir / "parameter_evolution.pdf"
    plot_parameter_evolution(parameter_pdf, param_histories, cell_targets, losses)
    pc1_pdf = args.out_dir / "parameter_pc1_evolution.pdf"
    pc1_loadings_csv = args.out_dir / "parameter_pc1_loadings.csv"
    plot_parameter_pc1_evolution(pc1_pdf, pc1_loadings_csv, param_histories, cell_targets, n_batch, n_cell, losses)

    summary_csv = args.out_dir / "raster_loss_summary.csv"
    write_loss_summary(summary_csv, cell_targets, psth_loss, cv_loss, sim_cv, data_cv, combined_rank, best)

    plot_raster_pdf(
        args.out_dir / "raster_comparisons_sorted_by_psth.pdf",
        output,
        data_raster,
        sim_psth,
        data_psth,
        bin_centers,
        cell_targets,
        psth_loss,
        cv_loss,
        sim_cv,
        data_cv,
        combined_rank,
        best,
        config,
        sort_key="psth",
        max_pages=args.max_raster_pages,
    )
    plot_raster_pdf(
        args.out_dir / "raster_comparisons_sorted_by_cv.pdf",
        output,
        data_raster,
        sim_psth,
        data_psth,
        bin_centers,
        cell_targets,
        psth_loss,
        cv_loss,
        sim_cv,
        data_cv,
        combined_rank,
        best,
        config,
        sort_key="cv",
        max_pages=args.max_raster_pages,
    )
    plot_raster_pdf(
        args.out_dir / "raster_comparisons_sorted_by_combined.pdf",
        output,
        data_raster,
        sim_psth,
        data_psth,
        bin_centers,
        cell_targets,
        psth_loss,
        cv_loss,
        sim_cv,
        data_cv,
        combined_rank,
        best,
        config,
        sort_key="combined",
        max_pages=args.max_raster_pages,
    )

    print(f"Wrote parameter evolution PDF: {parameter_pdf}")
    print(f"Wrote parameter PC1 evolution PDF: {pc1_pdf}")
    print(f"Wrote parameter PC1 loadings: {pc1_loadings_csv}")
    print(f"Wrote raster summaries: {summary_csv}")
    print_best("PSTH", best["psth"], cell_targets, psth_loss, cv_loss, sim_cv, data_cv)
    print_best("CV", best["cv"], cell_targets, psth_loss, cv_loss, sim_cv, data_cv)
    print_best("Combined rank", best["combined"], cell_targets, psth_loss, cv_loss, sim_cv, data_cv)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        "data_path": REPO_ROOT / config["paths"]["data"],
        "dt_ms": float(config["simulation"]["dt"]),
        "psth_granularity": int(config["simulation"]["PSTH_granularity"]),
        "data_target": config["simulation"].get("data_target", "peak"),
        "cell_targets": list(config["simulation"].get("cell_targets", [])),
    }


def normalize_output(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 3:
        output = output[:, :, None, :]
    if output.ndim != 4:
        raise ValueError(f"Expected output shape batch x trial x cell x time. Got {output.shape}.")
    return output.astype(bool)


def normalize_cell_targets(config: dict, n_cell: int) -> list[int]:
    cell_targets = [int(x) for x in config.get("cell_targets", [])]
    if len(cell_targets) != n_cell:
        print(f"Warning: config has {len(cell_targets)} cell target(s), but output has {n_cell}; using 1..{n_cell}.")
        return list(range(1, n_cell + 1))
    return cell_targets


def build_data_raster(config: dict, cell_targets: list[int], n_trial: int) -> np.ndarray:
    mat = sio.loadmat(config["data_path"], variable_names=["all_data"], squeeze_me=True, struct_as_record=False)
    all_data = np.ravel(mat["all_data"])
    sim_len = int(config["sim_len"])
    dt_seconds = config["dt_ms"] / 1000.0
    data_raster = np.zeros((len(cell_targets), n_trial, sim_len), dtype=bool)

    for cell_index, cell_target in enumerate(cell_targets):
        unit = all_data[cell_target - 1]
        angle_index = peak_angle_index(unit)
        timestamps = unit.ctrl_tar1_timestamps
        for trial_index in range(n_trial):
            spikes = get_trial_spikes(timestamps, trial_index, angle_index)
            spikes = spikes[(spikes >= 0) & (spikes <= sim_len * dt_seconds)]
            spike_index = np.floor(spikes / dt_seconds).astype(int)
            spike_index = spike_index[(spike_index >= 0) & (spike_index < sim_len)]
            data_raster[cell_index, trial_index, spike_index] = True
    return data_raster


def peak_angle_index(unit) -> int:
    tuning_type = str(unit.tuning_type).lower()
    if "contra" in tuning_type:
        return 0
    if "45" in tuning_type:
        return 1
    if "center" in tuning_type:
        return 2
    return 3


def get_trial_spikes(timestamps, trial_index: int, angle_index: int) -> np.ndarray:
    value = timestamps[trial_index, angle_index]
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.item() if value.size == 1 else value
    return np.ravel(np.asarray(value, dtype=float))


def raster_to_psth(raster: np.ndarray, granularity: int, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    n_batch, n_cell, n_trial, sim_len = raster.shape[0], raster.shape[2], raster.shape[1], raster.shape[3]
    n_bin = sim_len // granularity
    trimmed = raster[:, :, :, : n_bin * granularity]
    counts = trimmed.reshape(n_batch, n_trial, n_cell, n_bin, granularity).sum(axis=(1, 4))
    bin_edges = np.arange(n_bin + 1) * granularity * dt_ms / 1000.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts.astype(float), bin_centers


def cv_from_raster(raster: np.ndarray) -> float:
    isis = []
    trials_with_spikes = 0
    for trial_index in range(raster.shape[0]):
        spike_index = np.flatnonzero(raster[trial_index])
        if spike_index.size > 0:
            trials_with_spikes += 1
        if spike_index.size > 1:
            isis.append(np.diff(spike_index).astype(float))
    if trials_with_spikes <= 1 or not isis:
        return np.nan
    all_isis = np.concatenate(isis)
    if all_isis.size <= 1 or np.mean(all_isis) == 0:
        return np.nan
    return float(np.std(all_isis, ddof=1) / np.mean(all_isis))


def get_params(output_path: Path):
    mat = sio.loadmat(output_path, squeeze_me=True, struct_as_record=False)
    return mat["params"]


def ordered_param_names(params) -> list[str]:
    fields = list(getattr(params, "_fieldnames", []))
    ordered = [name for name in PARAMETER_ORDER if name in fields]
    ordered.extend(name for name in fields if name not in ordered)
    return ordered


def normalize_param_history(value: np.ndarray, n_batch: int, n_cell: int, n_epochs: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        return arr.reshape(1, 1, 1)
    if arr.ndim == 1:
        if arr.size == n_epochs:
            return arr.reshape(1, 1, n_epochs)
        if arr.size == n_batch:
            return arr.reshape(n_batch, 1, 1)
        return arr.reshape(1, 1, arr.size)
    if arr.ndim == 2:
        if arr.shape == (n_batch, n_epochs):
            return arr[:, None, :]
        if arr.shape == (n_cell, n_epochs):
            return arr[None, :, :]
        if arr.shape == (n_batch, n_cell):
            return arr[:, :, None]
        return arr[:, None, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Cannot normalize parameter history with shape {arr.shape}.")


def plot_parameter_evolution(pdf_path: Path, param_histories: dict[str, np.ndarray], cell_targets: list[int], losses: np.ndarray) -> None:
    colors = plt.get_cmap("tab10")
    with PdfPages(pdf_path) as pdf:
        if losses.size:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(np.arange(losses.size), losses, color="black", linewidth=2)
            ax.set_title("Saved Mean SSE Loss By Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean SSE")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for param_index, (param_name, history) in enumerate(param_histories.items()):
            n_batch, n_cell, n_epochs = history.shape
            for page_start in range(0, n_cell, 4):
                page_cells = list(range(page_start, min(page_start + 4, n_cell)))
                fig, axes = plt.subplots(len(page_cells), 1, figsize=(11, 3.1 * len(page_cells)), squeeze=False)
                color = colors(param_index % 10)
                epochs = np.arange(n_epochs)

                for ax, cell_index in zip(axes.ravel(), page_cells):
                    traces = history[:, cell_index, :]
                    mean_trace = np.nanmean(traces, axis=0)
                    std_trace = np.nanstd(traces, axis=0)

                    ax.plot(epochs, traces.T, color=color, alpha=0.16, linewidth=0.8)
                    ax.fill_between(
                        epochs,
                        mean_trace - std_trace,
                        mean_trace + std_trace,
                        color=color,
                        alpha=0.20,
                        linewidth=0,
                        label="mean +/- 1 std",
                    )
                    ax.plot(epochs, mean_trace, color=color, linewidth=3.0, label="batch mean")
                    ax.set_title(f"{param_name} | cell target {cell_targets[cell_index]}")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel(param_name)
                    ax.grid(alpha=0.25)
                    ax.legend(loc="best")

                fig.suptitle(f"Parameter Evolution: {param_name}", fontsize=14, fontweight="bold")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def plot_parameter_pc1_evolution(
    pdf_path: Path,
    loadings_csv: Path,
    param_histories: dict[str, np.ndarray],
    cell_targets: list[int],
    n_batch: int,
    n_cell: int,
    losses: np.ndarray,
) -> None:
    n_epochs = min(history.shape[2] for history in param_histories.values())
    if losses.size:
        n_epochs = min(n_epochs, losses.size)
    param_names = list(param_histories.keys())
    loadings_rows = []

    with PdfPages(pdf_path) as pdf:
        for cell_index in range(n_cell):
            cube = parameter_cube_for_cell(param_histories, param_names, n_batch, n_cell, n_epochs, cell_index)
            scores, loadings, explained_ratio = first_pc_projection(cube)
            epochs = np.arange(n_epochs)
            mean_score = np.nanmean(scores, axis=0)
            std_score = np.nanstd(scores, axis=0)

            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2.1, 1.0]})

            axes[0].plot(epochs, scores.T, color="#4c78a8", alpha=0.16, linewidth=0.8)
            axes[0].fill_between(
                epochs,
                mean_score - std_score,
                mean_score + std_score,
                color="#4c78a8",
                alpha=0.20,
                linewidth=0,
                label="mean +/- 1 std",
            )
            axes[0].plot(epochs, mean_score, color="#1f4e79", linewidth=3.0, label="batch mean")
            axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.35)
            axes[0].set_title(
                f"PC1 trajectory over epochs | cell target {cell_targets[cell_index]} | "
                f"{explained_ratio * 100:.1f}% variance"
            )
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("PC1 score")
            axes[0].grid(alpha=0.25)
            axes[0].legend(loc="best")

            order = np.argsort(np.abs(loadings))[::-1]
            sorted_names = [param_names[i] for i in order]
            sorted_loadings = loadings[order]
            colors = ["#1f77b4" if value >= 0 else "#d62728" for value in sorted_loadings]
            axes[1].bar(np.arange(len(sorted_loadings)), sorted_loadings, color=colors, alpha=0.85)
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].set_xticks(np.arange(len(sorted_names)))
            axes[1].set_xticklabels(sorted_names, rotation=35, ha="right")
            axes[1].set_ylabel("PC1 loading")
            axes[1].set_title("PC1 parameter loadings, standardized parameter units")
            axes[1].grid(axis="y", alpha=0.25)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            for param_name, loading in zip(param_names, loadings):
                loadings_rows.append(
                    {
                        "cell_target": cell_targets[cell_index],
                        "parameter": param_name,
                        "pc1_loading": loading,
                        "explained_variance_ratio": explained_ratio,
                    }
                )

    with loadings_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cell_target", "parameter", "pc1_loading", "explained_variance_ratio"],
        )
        writer.writeheader()
        writer.writerows(loadings_rows)


def parameter_cube_for_cell(
    param_histories: dict[str, np.ndarray],
    param_names: list[str],
    n_batch: int,
    n_cell: int,
    n_epochs: int,
    cell_index: int,
) -> np.ndarray:
    traces = []
    for param_name in param_names:
        history = broadcast_history(param_histories[param_name], n_batch, n_cell, n_epochs)
        traces.append(history[:, cell_index, :])
    return np.stack(traces, axis=2)


def broadcast_history(history: np.ndarray, n_batch: int, n_cell: int, n_epochs: int) -> np.ndarray:
    history = history[:, :, :n_epochs]
    if history.shape[0] == 1 and n_batch > 1:
        history = np.broadcast_to(history, (n_batch, history.shape[1], history.shape[2]))
    if history.shape[1] == 1 and n_cell > 1:
        history = np.broadcast_to(history, (history.shape[0], n_cell, history.shape[2]))
    if history.shape[0] != n_batch or history.shape[1] != n_cell:
        raise ValueError(f"Cannot broadcast parameter history shape {history.shape} to ({n_batch}, {n_cell}, {n_epochs}).")
    return history


def first_pc_projection(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n_batch, n_epochs, n_param = cube.shape
    matrix = cube.reshape(n_batch * n_epochs, n_param)
    feature_mean = np.nanmean(matrix, axis=0)
    feature_std = np.nanstd(matrix, axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    standardized = (matrix - feature_mean) / feature_std
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    loadings = vt[0]
    scores = (standardized @ loadings).reshape(n_batch, n_epochs)
    variance = singular_values**2
    explained_ratio = float(variance[0] / variance.sum()) if variance.sum() > 0 else float("nan")

    mean_score = np.nanmean(scores, axis=0)
    if mean_score[-1] < mean_score[0]:
        scores = -scores
        loadings = -loadings
    return scores, loadings, explained_ratio


def write_loss_summary(
    csv_path: Path,
    cell_targets: list[int],
    psth_loss: np.ndarray,
    cv_loss: np.ndarray,
    sim_cv: np.ndarray,
    data_cv: np.ndarray,
    combined_rank: np.ndarray,
    best: dict[str, tuple[int, int]],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "batch",
                "cell_index",
                "cell_target",
                "psth_mse",
                "cv_loss",
                "sim_cv",
                "data_cv",
                "combined_rank",
                "is_best_psth",
                "is_best_cv",
                "is_best_combined",
            ]
        )
        for batch_index in range(psth_loss.shape[0]):
            for cell_index, cell_target in enumerate(cell_targets):
                writer.writerow(
                    [
                        batch_index + 1,
                        cell_index + 1,
                        cell_target,
                        psth_loss[batch_index, cell_index],
                        cv_loss[batch_index, cell_index],
                        sim_cv[batch_index, cell_index],
                        data_cv[cell_index],
                        combined_rank[batch_index, cell_index],
                        (batch_index, cell_index) == best["psth"],
                        (batch_index, cell_index) == best["cv"],
                        (batch_index, cell_index) == best["combined"],
                    ]
                )


def plot_raster_pdf(
    pdf_path: Path,
    output: np.ndarray,
    data_raster: np.ndarray,
    sim_psth: np.ndarray,
    data_psth: np.ndarray,
    bin_centers: np.ndarray,
    cell_targets: list[int],
    psth_loss: np.ndarray,
    cv_loss: np.ndarray,
    sim_cv: np.ndarray,
    data_cv: np.ndarray,
    combined_rank: np.ndarray,
    best: dict[str, tuple[int, int]],
    config: dict,
    sort_key: str,
    max_pages: int,
) -> None:
    candidates = [(batch_index, cell_index) for batch_index in range(output.shape[0]) for cell_index in range(output.shape[2])]
    if sort_key == "psth":
        candidates.sort(key=lambda item: np.nan_to_num(psth_loss[item], nan=np.inf))
    elif sort_key == "cv":
        candidates.sort(key=lambda item: np.nan_to_num(cv_loss[item], nan=np.inf))
    elif sort_key == "combined":
        candidates.sort(key=lambda item: combined_rank[item])
    else:
        raise ValueError(f"Unknown sort key {sort_key}.")

    with PdfPages(pdf_path) as pdf:
        for page_index, (batch_index, cell_index) in enumerate(candidates[:max_pages]):
            fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1.0, 1.0, 1.15]})
            plot_raster(axes[0], output[batch_index, :, cell_index, :], config["dt_ms"], "Simulation raster")
            plot_raster(axes[1], data_raster[cell_index], config["dt_ms"], "Data raster")

            axes[2].plot(bin_centers, data_psth[cell_index], color="black", linewidth=1.8, label="Data")
            axes[2].plot(bin_centers, sim_psth[batch_index, cell_index], color="#d62728", linewidth=1.4, label="Simulation")
            axes[2].set_xlim(0, output.shape[-1] * config["dt_ms"] / 1000.0)
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel(f"Spikes / {config['psth_granularity'] * config['dt_ms']:.3g} ms bin")
            axes[2].set_title("PSTH overlay")
            axes[2].grid(alpha=0.25)
            axes[2].legend(loc="best")

            title, color = title_for_candidate(
                batch_index,
                cell_index,
                cell_targets,
                psth_loss,
                cv_loss,
                sim_cv,
                data_cv,
                combined_rank,
                best,
                sort_key,
                page_index,
            )
            fig.suptitle(title, color=color, fontsize=10, fontweight="bold", y=0.985, linespacing=1.25)
            fig.tight_layout(rect=[0, 0, 1, 0.88])
            pdf.savefig(fig)
            plt.close(fig)


def plot_raster(ax, raster: np.ndarray, dt_ms: float, title: str) -> None:
    trial_index, time_index = np.where(raster)
    ax.scatter(time_index * dt_ms / 1000.0, trial_index + 1, s=5, color="#1f77b4", alpha=0.85)
    ax.set_xlim(0, raster.shape[1] * dt_ms / 1000.0)
    ax.set_ylim(0.5, raster.shape[0] + 0.5)
    ax.invert_yaxis()
    ax.set_ylabel("Trial")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)


def title_for_candidate(
    batch_index: int,
    cell_index: int,
    cell_targets: list[int],
    psth_loss: np.ndarray,
    cv_loss: np.ndarray,
    sim_cv: np.ndarray,
    data_cv: np.ndarray,
    combined_rank: np.ndarray,
    best: dict[str, tuple[int, int]],
    sort_key: str,
    page_index: int,
) -> tuple[str, str]:
    tags = []
    color = "black"
    if (batch_index, cell_index) == best["psth"]:
        tags.append("BEST PSTH")
        color = "forestgreen"
    if (batch_index, cell_index) == best["cv"]:
        tags.append("BEST CV")
        color = "darkorange" if color == "black" else "purple"
    if (batch_index, cell_index) == best["combined"]:
        tags.append("BEST COMBINED")
        color = "purple" if tags else "black"
    tag_text = f" | {' + '.join(tags)}" if tags else ""
    return (
        f"{sort_key.upper()} rank {page_index + 1} | cell {cell_targets[cell_index]} | batch {batch_index + 1}{tag_text}\n"
        f"PSTH MSE {psth_loss[batch_index, cell_index]:.4g} | CV loss {cv_loss[batch_index, cell_index]:.4g} | "
        f"combined rank {combined_rank[batch_index, cell_index]:.0f}\n"
        f"sim CV {sim_cv[batch_index, cell_index]:.4g} | data CV {data_cv[cell_index]:.4g}",
        color,
    )


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks_out = np.empty_like(order, dtype=float)
    ranks_out[order] = np.arange(1, values.size + 1, dtype=float)
    return ranks_out


def print_best(
    name: str,
    index: tuple[int, int],
    cell_targets: list[int],
    psth_loss: np.ndarray,
    cv_loss: np.ndarray,
    sim_cv: np.ndarray,
    data_cv: np.ndarray,
) -> None:
    batch_index, cell_index = index
    print(
        f"Best {name}: cell {cell_targets[cell_index]}, batch {batch_index + 1}, "
        f"PSTH MSE {psth_loss[batch_index, cell_index]:.4g}, "
        f"CV loss {cv_loss[batch_index, cell_index]:.4g}, "
        f"sim CV {sim_cv[batch_index, cell_index]:.4g}, data CV {data_cv[cell_index]:.4g}"
    )


if __name__ == "__main__":
    main()
