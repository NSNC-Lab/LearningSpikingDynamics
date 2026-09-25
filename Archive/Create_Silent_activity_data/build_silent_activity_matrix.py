from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


COLUMNS = [
    "activity_angle_1_hz",
    "activity_angle_2_hz",
    "activity_angle_3_hz",
    "activity_angle_4_hz",
    "peak_angle",
]

TUNING_TO_ANGLE = {
    "contra-tuned": 0,
    "45\u00b0-tuned": 1,
    "center-tuned": 2,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def peak_angle_from_tuning(tuning_type: object) -> int:
    return TUNING_TO_ANGLE.get(str(tuning_type).strip(), 3)


def count_silent_spikes(spike_times_by_trial: np.ndarray) -> np.ndarray:
    counts = np.frompyfunc(
        lambda x: np.count_nonzero(np.asarray(x, dtype=float).ravel() < 0.0),
        1,
        1,
    )(spike_times_by_trial)
    return counts.astype(float)


def silent_activity_hz(spike_times_by_trial: np.ndarray, silent_duration_s: float) -> np.ndarray:
    if spike_times_by_trial.ndim != 2:
        raise ValueError(f"Expected a trials x angles timestamp array, got {spike_times_by_trial.shape}.")
    if silent_duration_s <= 0:
        raise ValueError("silent_duration_s must be positive.")

    counts = count_silent_spikes(spike_times_by_trial)
    trials_per_angle = spike_times_by_trial.shape[0]
    return counts.sum(axis=0) / (trials_per_angle * silent_duration_s)



def build_matrix(all_data: np.ndarray, all_data2: np.ndarray, timestamp_field: str, timestamp_field2: str, silent_duration_s: float) -> np.ndarray:
    matrix = np.full((len(all_data) + len(all_data2), len(COLUMNS)), np.nan, dtype=float)
    for cell_idx, unit in enumerate(all_data):
        spike_times_by_trial = getattr(unit, timestamp_field)
        activity = silent_activity_hz(spike_times_by_trial, silent_duration_s)
        if activity.shape[0] != 4:
            raise ValueError(f"Cell {cell_idx} has {activity.shape[0]} angles, expected 4.")
        matrix[cell_idx, :4] = activity
        matrix[cell_idx, 4] = peak_angle_from_tuning(unit.tuning_type)

    for cell_idx, unit in enumerate(all_data2):
            spike_times_by_trial = getattr(unit, timestamp_field2)
            activity = silent_activity_hz(spike_times_by_trial, silent_duration_s)
            if activity.shape[0] != 4:
                raise ValueError(f"Cell {cell_idx} in dataset2 has {activity.shape[0]} angles, expected 4.")
            matrix[cell_idx + len(all_data), :4] = activity
            matrix[cell_idx + len(all_data), 4] = peak_angle_from_tuning(unit.tuning_type)
    return matrix


def write_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(matrix)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Build a silent-period activity matrix for all cells.")
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=root / "Data" / "Data" / "all_units_info_with_polished_criteria_modified_perf.mat",
    )

    parser.add_argument(
            "--mat-path2",
            type=Path,
            default=root / "Data" / "Data" / "all_cluster_info_atten_modMartin_OliverCriterion.mat",
    )

    parser.add_argument("--timestamp-field", default="ctrl_tar1_timestamps")
    parser.add_argument("--timestamp-field2", default="passive_tar1_timestamps")
    parser.add_argument("--silent-duration-s", type=float, default=1.0)
    parser.add_argument("--out-npy", type=Path, default=root / "Archive" / "silent_activity_matrix.npy")
    parser.add_argument("--out-csv", type=Path, default=root / "Archive" / "silent_activity_matrix.csv")
    parser.add_argument("--out-json", type=Path, default=root / "Archive" / "silent_activity_matrix_columns.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mat = loadmat(args.mat_path, variable_names=["all_data"], squeeze_me=True, struct_as_record=False)
    mat2 = loadmat(args.mat_path2, variable_names=["all_data"], squeeze_me=True, struct_as_record=False)
    matrix = build_matrix(mat["all_data"], mat2["all_data"], args.timestamp_field, args.timestamp_field2, args.silent_duration_s)

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, matrix)
    write_csv(args.out_csv, matrix)
    args.out_json.write_text(json.dumps({"columns": COLUMNS}, indent=2), encoding="utf-8")

    print(f"Saved {matrix.shape} matrix to {args.out_npy}")
    print(f"Columns: {', '.join(COLUMNS)}")


if __name__ == "__main__":
    main()
