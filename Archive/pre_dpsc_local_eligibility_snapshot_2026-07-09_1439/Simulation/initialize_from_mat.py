"""Load initial learnable parameters from a saved MATLAB output file.

This module is intentionally optional: Parameter_initialization.py calls it
only when the config requests a .mat-backed initialization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat


MAT_STRUCT_FIELD_TO_PARAM = {
    "strf_gain": "Strf_gain",
    "strf_alpha": "Strf_alpha",
    "output_ad": "output_ad",
    "abs_ref": "abs_ref",
    "rel_ref_a": "rel_ref_a",
    "rel_ref_b": "rel_ref_b",
    "rel_ref_c": "rel_ref_c",
    "on_ron_gsyn": "on_ron_gSYN",
    "off_ron_gsyn": "off_ron_gSYN",
    "sonoff_ron_gsyn": "sonoff_ron_gSYN",
    "on_sonoff_gsyn": "on_sonoff_gSYN",
    "off_sonoff_gsyn": "off_sonoff_gSYN",
}

DEFAULT_COMPACT_PARAM_ORDER = [
    "Strf_gain",
    "Strf_alpha",
    "output_ad",
    "on_ron_gSYN",
    "off_ron_gSYN",
    "on_sonoff_gSYN",
    "off_sonoff_gSYN",
    "sonoff_ron_gSYN",
]


def get_mat_initialization_config(args: dict[str, Any]) -> dict[str, Any] | None:
    """Return the optional .mat initialization config, or None if disabled.

    Preferred config shape:

    parameter_initialization:
      from_mat:
        enabled: true
        path: "C:/path/to/output.mat"

    A shorter legacy-friendly shape is also accepted:

    simulation:
      init_from_mat: "C:/path/to/output.mat"
    """
    init_cfg = args.get("parameter_initialization", {}).get("from_mat")
    if init_cfg is None:
        init_cfg = args.get("simulation", {}).get("init_from_mat")

    if init_cfg in (None, False):
        return None

    if isinstance(init_cfg, str):
        return {"enabled": True, "path": init_cfg}

    if isinstance(init_cfg, dict):
        if not init_cfg.get("enabled", False):
            return None
        return init_cfg

    raise TypeError(
        "MAT initialization config must be a path string, a dict, False, or omitted."
    )


def resolve_mat_path(path_value: str | Path) -> Path:
    mat_path = Path(path_value).expanduser()
    if not mat_path.is_absolute():
        mat_path = Path.cwd() / mat_path
    return mat_path.resolve()


def find_unique_axis(shape: tuple[int, ...], size: int, role: str) -> int:
    matches = [axis for axis, axis_size in enumerate(shape) if axis_size == size]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {role} axis of size {size}, "
            f"found {matches} in shape {shape}."
        )
    return matches[0]


def normalize_named_param_array(
    value: np.ndarray,
    param_name: str,
    expected_batches: int,
    expected_cells: int,
) -> np.ndarray:
    """Normalize named parameter tracker arrays to batch x cell.

    Saved current-project trackers are batch x cell x epoch. Extra axes are
    treated as tracker/epoch axes and the final index is used.
    """
    arr = np.asarray(value, dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape == (expected_batches, expected_cells):
            return np.ascontiguousarray(arr)
        if arr.shape == (expected_cells, expected_batches):
            return np.ascontiguousarray(arr.T)

    if arr.ndim >= 3 and arr.shape[0] == expected_batches and arr.shape[1] == expected_cells:
        while arr.ndim > 2:
            arr = np.take(arr, indices=-1, axis=-1)
        return np.ascontiguousarray(arr)

    arr = np.squeeze(arr)

    if arr.ndim < 2:
        raise ValueError(f"{param_name}: expected at least 2D, got shape {arr.shape}")

    if arr.ndim == 2 and expected_cells == 1:
        batch_axes = [axis for axis, axis_size in enumerate(arr.shape) if axis_size == expected_batches]
        if len(batch_axes) == 1:
            batch_axis = batch_axes[0]
            epoch_axis = 1 - batch_axis
            arr = np.take(arr, indices=-1, axis=epoch_axis)
            if batch_axis == 1:
                arr = arr.T
            return np.ascontiguousarray(arr.reshape(expected_batches, 1))

    while arr.ndim > 2:
        batch_axis = find_unique_axis(arr.shape, expected_batches, "batch")
        cell_axis = find_unique_axis(arr.shape, expected_cells, "cell")
        extra_axes = [
            axis for axis in range(arr.ndim) if axis not in {batch_axis, cell_axis}
        ]
        if not extra_axes:
            raise ValueError(
                f"{param_name}: shape {arr.shape} has extra dimensions but no "
                "epoch/tracker axis could be identified."
            )
        arr = np.take(arr, indices=-1, axis=extra_axes[-1])

    batch_axis = find_unique_axis(arr.shape, expected_batches, "batch")
    cell_axis = find_unique_axis(arr.shape, expected_cells, "cell")
    return np.ascontiguousarray(np.transpose(arr, (batch_axis, cell_axis)))


def extract_mat_struct_params(
    params_var: np.ndarray,
    expected_batches: int,
    expected_cells: int,
) -> dict[str, np.ndarray] | None:
    """Extract named params from MATLAB structs saved by output_handler.py."""
    if not isinstance(params_var, np.ndarray) or params_var.dtype != object:
        return None

    mat_struct = params_var.ravel()[0]
    field_names = getattr(mat_struct, "_fieldnames", None)
    if not field_names:
        return None

    loaded: dict[str, np.ndarray] = {}
    for field_name in field_names:
        key = MAT_STRUCT_FIELD_TO_PARAM.get(field_name.lower())
        if key is None:
            continue
        loaded[key] = normalize_named_param_array(
            getattr(mat_struct, field_name), key, expected_batches, expected_cells
        )

    return loaded


def normalize_compact_params(
    params_var: np.ndarray,
    expected_batches: int,
    expected_cells: int,
    compact_param_order: list[str],
) -> np.ndarray:
    """Normalize compact old-model params to param x cell x batch."""
    arr = np.asarray(params_var, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim < 3:
        raise ValueError(
            "Compact MAT params must have param, cell, and batch axes; "
            f"got shape {arr.shape}."
        )

    while arr.ndim > 3:
        param_axis = find_unique_axis(arr.shape, len(compact_param_order), "param")
        batch_axis = find_unique_axis(arr.shape, expected_batches, "batch")
        cell_axis = find_unique_axis(arr.shape, expected_cells, "cell")
        extra_axes = [
            axis
            for axis in range(arr.ndim)
            if axis not in {param_axis, batch_axis, cell_axis}
        ]
        if not extra_axes:
            raise ValueError(
                f"Compact params shape {arr.shape} has extra dimensions but no "
                "epoch/tracker axis could be identified."
            )
        arr = np.take(arr, indices=-1, axis=extra_axes[-1])

    param_axis = find_unique_axis(arr.shape, len(compact_param_order), "param")
    batch_axis = find_unique_axis(arr.shape, expected_batches, "batch")
    cell_axis = find_unique_axis(arr.shape, expected_cells, "cell")
    return np.ascontiguousarray(np.transpose(arr, (param_axis, cell_axis, batch_axis)))


def extract_compact_params(
    params_var: np.ndarray,
    expected_batches: int,
    expected_cells: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Extract params from old compact arrays shaped epoch x param x cell x batch."""
    compact_param_order = list(config.get("compact_param_order", DEFAULT_COMPACT_PARAM_ORDER))
    compact = normalize_compact_params(
        params_var, expected_batches, expected_cells, compact_param_order
    )

    strf_alpha_seconds = bool(config.get("compact_strf_alpha_seconds", True))
    loaded: dict[str, np.ndarray] = {}

    for param_index, param_name in enumerate(compact_param_order):
        values = compact[param_index, :, :].T
        if param_name == "Strf_alpha" and strf_alpha_seconds:
            values = values * 1000.0
        loaded[param_name] = np.ascontiguousarray(values, dtype=np.float32)

    return loaded


def load_last_epoch_params_from_mat(
    mat_path: str | Path,
    args: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Load last-epoch learnable parameters from a MATLAB output/checkpoint file."""
    config = config or {}
    resolved_path = resolve_mat_path(mat_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"MAT initialization file does not exist: {resolved_path}")

    expected_batches = int(args["simulation"]["batch_size"])
    expected_cells = len(args["simulation"]["cell_targets"])

    mat = loadmat(resolved_path, squeeze_me=False, struct_as_record=False)
    if "params" not in mat and "checkpoint_params" not in mat:
        raise KeyError(
            f"{resolved_path} does not contain a 'params' or 'checkpoint_params' variable. "
            f"Available keys: {[key for key in mat if not key.startswith('__')]}"
        )

    if "checkpoint_params" in mat and bool(config.get("prefer_checkpoint_params", True)):
        checkpoint_params = extract_mat_struct_params(
            mat["checkpoint_params"], expected_batches, expected_cells
        )
        if checkpoint_params is not None:
            return checkpoint_params

    if "params" not in mat:
        raise KeyError(
            f"{resolved_path} does not contain a 'params' variable and checkpoint_params "
            "could not be read."
        )

    params_var = mat["params"]
    named_params = extract_mat_struct_params(
        params_var, expected_batches, expected_cells
    )
    if named_params is not None:
        return named_params

    return extract_compact_params(params_var, expected_batches, expected_cells, config)


def normalize_adam_array(
    value: np.ndarray,
    name: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float32)
    if raw.shape == expected_shape:
        return np.ascontiguousarray(raw)

    arr = np.squeeze(raw)
    expected_batches, expected_cells, expected_params = expected_shape
    if expected_cells == 1 and arr.shape == (expected_batches, expected_params):
        return np.ascontiguousarray(arr[:, None, :])
    if expected_batches == 1 and arr.shape == (expected_cells, expected_params):
        return np.ascontiguousarray(arr[None, :, :])

    if arr.shape != expected_shape:
        raise ValueError(
            f"Adam {name}: loaded shape {arr.shape} does not match expected "
            f"shape {expected_shape}."
        )
    return np.ascontiguousarray(arr)


def extract_adam_state(
    adam_var: np.ndarray,
    expected_batches: int,
    expected_cells: int,
    expected_params: int,
) -> dict[str, np.ndarray | int] | None:
    if not isinstance(adam_var, np.ndarray) or adam_var.dtype != object:
        return None

    mat_struct = adam_var.ravel()[0]
    field_names = getattr(mat_struct, "_fieldnames", None)
    if not field_names or not {"m", "v", "t"}.issubset(set(field_names)):
        return None

    expected_shape = (expected_batches, expected_cells, expected_params)
    return {
        "m": normalize_adam_array(getattr(mat_struct, "m"), "m", expected_shape),
        "v": normalize_adam_array(getattr(mat_struct, "v"), "v", expected_shape),
        "t": int(np.asarray(getattr(mat_struct, "t")).squeeze()),
    }


def load_adam_state_from_mat(
    mat_path: str | Path,
    args: dict[str, Any],
) -> dict[str, np.ndarray | int] | None:
    resolved_path = resolve_mat_path(mat_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"MAT initialization file does not exist: {resolved_path}")

    mat = loadmat(resolved_path, squeeze_me=False, struct_as_record=False)
    if "adam" not in mat:
        return None

    return extract_adam_state(
        mat["adam"],
        int(args["simulation"]["batch_size"]),
        len(args["simulation"]["cell_targets"]),
        int(args["simulation"]["num_params"]),
    )


def maybe_replace_initialization_from_mat(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Replace initialized params from a configured .mat file when enabled."""
    config = get_mat_initialization_config(args)
    if config is None:
        return params

    if "path" not in config:
        raise KeyError(
            "MAT initialization is enabled, but no path was provided. "
            "Use parameter_initialization.from_mat.path."
        )

    loaded = load_last_epoch_params_from_mat(config["path"], args, config)
    strict = bool(config.get("strict", True))

    missing = [key for key in loaded if key not in params]
    if missing and strict:
        raise KeyError(f"MAT file contained parameter(s) not used here: {missing}")

    replaced = []
    skipped = []
    for key, value in loaded.items():
        if key not in params:
            skipped.append(key)
            continue
        if params[key].shape != value.shape:
            raise ValueError(
                f"{key}: loaded shape {value.shape} does not match initialized "
                f"shape {params[key].shape}."
            )
        params[key] = value.astype(np.float32, copy=False)
        replaced.append(key)

    print(
        "Initialized from MAT last epoch: "
        f"{resolve_mat_path(config['path'])} "
        f"({len(replaced)} replaced: {', '.join(replaced)})"
    )
    if skipped:
        print(f"Skipped MAT parameter(s) not present in this model: {', '.join(skipped)}")

    return params


def maybe_restore_adam_from_mat(
    args: dict[str, Any],
    states: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Restore Adam m/v/t from a configured checkpoint when present."""
    config = get_mat_initialization_config(args)
    if config is None or not bool(config.get("load_adam", True)):
        return states

    loaded = load_adam_state_from_mat(config["path"], args)
    if loaded is None:
        print(
            "MAT checkpoint did not contain Adam state; starting Adam from zeros."
        )
        return states

    import torch

    states["neurons"]["Adam"]["m"] = torch.tensor(
        loaded["m"], device=device, dtype=torch.float32
    )
    states["neurons"]["Adam"]["v"] = torch.tensor(
        loaded["v"], device=device, dtype=torch.float32
    )
    states["neurons"]["Adam"]["t"] = int(loaded["t"])
    print(
        "Restored Adam state from MAT checkpoint: "
        f"t={states['neurons']['Adam']['t']}"
    )
    return states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect parameters that would be loaded from a .mat file."
    )
    parser.add_argument("mat_path", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("simulation_config.yaml"),
        help="Simulation config used for expected batch/cell dimensions.",
    )
    args_cli = parser.parse_args()

    with args_cli.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    loaded = load_last_epoch_params_from_mat(args_cli.mat_path, config)
    for key, value in loaded.items():
        print(
            f"{key}: shape={value.shape}, "
            f"min={np.nanmin(value):.6g}, max={np.nanmax(value):.6g}"
        )


if __name__ == "__main__":
    main()
