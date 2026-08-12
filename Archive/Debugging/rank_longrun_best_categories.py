from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from analyze_simulation_output import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    build_data_raster,
    cv_from_raster,
    load_config,
    ranks,
    raster_to_psth,
)


LONGRUN_MAT = REPO_ROOT / "Archive" / "LongRunResults.mat"
OUT_DIR = REPO_ROOT / "Archive" / "Debugging" / "longrun_ranking_analysis"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank LongRunResults candidates for selected cell targets.")
    parser.add_argument("--cell-targets", type=int, nargs="+", default=[7], help="1-based cell ids to analyze.")
    parser.add_argument("--longrun-mat", type=Path, default=LONGRUN_MAT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(DEFAULT_CONFIG)

    raw = sio.loadmat(args.longrun_mat, squeeze_me=False, struct_as_record=False)["output"]
    output = normalize_longrun_output(raw, args.cell_targets)
    n_batch, n_trial, n_cell, sim_len = output.shape

    config["sim_len"] = sim_len
    cell_targets = args.cell_targets
    data_raster = build_data_raster(config, cell_targets, n_trial)

    data_for_psth = data_raster.transpose(1, 0, 2)[None, :, :, :]
    data_psth, _ = raster_to_psth(data_for_psth, config["psth_granularity"], config["dt_ms"])
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

    psth_sortable = np.nan_to_num(psth_loss, nan=np.inf, posinf=np.inf, neginf=np.inf)
    cv_sortable = np.nan_to_num(cv_loss, nan=np.inf, posinf=np.inf, neginf=np.inf)
    psth_rank = ranks(psth_sortable.ravel()).reshape(psth_loss.shape)
    cv_rank = ranks(cv_sortable.ravel()).reshape(cv_loss.shape)
    combined_score = psth_rank + cv_rank
    combined_rank = ranks(combined_score.ravel()).reshape(combined_score.shape)

    best_indices = {
        "best_psth": tuple(np.unravel_index(np.argmin(psth_sortable), psth_loss.shape)),
        "best_cv": tuple(np.unravel_index(np.argmin(cv_sortable), cv_loss.shape)),
        "best_combined": tuple(np.unravel_index(np.argmin(combined_score), combined_score.shape)),
    }

    summary = {
        name: summarize_candidate(
            batch_index,
            cell_index,
            cell_targets,
            psth_loss,
            cv_loss,
            sim_cv,
            data_cv,
            psth_rank,
            cv_rank,
            combined_rank,
            combined_score,
        )
        for name, (batch_index, cell_index) in best_indices.items()
    }

    cell_suffix = "_".join(str(cell) for cell in cell_targets)
    csv_path = args.out_dir / f"longrun_best_category_rankings_cells_{cell_suffix}.csv"
    json_path = args.out_dir / f"longrun_best_category_rankings_cells_{cell_suffix}.json"
    write_summary_csv(csv_path, summary)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analyzed LongRunResults over {n_batch} batches x {n_cell} cells = {n_batch * n_cell} candidates")
    for name, item in summary.items():
        print(
            f"{name}: cell {item['cell_target']}, batch {item['batch_one_based']}, "
            f"PSTH rank {item['psth_rank']} / {item['total_candidates']}, "
            f"CV rank {item['cv_rank']} / {item['total_candidates']}, "
            f"combined rank {item['combined_rank']} / {item['total_candidates']}, "
            f"PSTH MSE {item['psth_mse']:.6g}, CV loss {item['cv_loss']:.6g}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def normalize_longrun_output(raw: np.ndarray, cell_targets: list[int]) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim != 5:
        raise ValueError(f"Expected LongRun output shape cell x batch x trial x output x time. Got {raw.shape}.")
    cell_indices = [cell_target - 1 for cell_target in cell_targets]
    output = raw[cell_indices, :, :, 0, :].transpose(1, 2, 0, 3)
    return output.astype(bool)


def summarize_candidate(
    batch_index: int,
    cell_index: int,
    cell_targets: list[int],
    psth_loss: np.ndarray,
    cv_loss: np.ndarray,
    sim_cv: np.ndarray,
    data_cv: np.ndarray,
    psth_rank: np.ndarray,
    cv_rank: np.ndarray,
    combined_rank: np.ndarray,
    combined_score: np.ndarray,
) -> dict:
    total = int(psth_loss.size)
    return {
        "cell_target": int(cell_targets[cell_index]),
        "batch_zero_based": int(batch_index),
        "batch_one_based": int(batch_index + 1),
        "psth_mse": float(psth_loss[batch_index, cell_index]),
        "cv_loss": float(cv_loss[batch_index, cell_index]),
        "sim_cv": float(sim_cv[batch_index, cell_index]),
        "data_cv": float(data_cv[cell_index]),
        "psth_rank": int(psth_rank[batch_index, cell_index]),
        "cv_rank": int(cv_rank[batch_index, cell_index]),
        "combined_score": float(combined_score[batch_index, cell_index]),
        "combined_rank": int(combined_rank[batch_index, cell_index]),
        "total_candidates": total,
    }


def write_summary_csv(csv_path: Path, summary: dict[str, dict]) -> None:
    columns = [
        "category",
        "cell_target",
        "batch_zero_based",
        "batch_one_based",
        "psth_mse",
        "cv_loss",
        "sim_cv",
        "data_cv",
        "psth_rank",
        "cv_rank",
        "combined_score",
        "combined_rank",
        "total_candidates",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for category, item in summary.items():
            row = {"category": category}
            row.update(item)
            writer.writerow(row)


if __name__ == "__main__":
    main()
