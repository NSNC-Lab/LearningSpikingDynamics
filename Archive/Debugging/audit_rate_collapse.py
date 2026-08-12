"""Audit a saved simulation for firing-rate collapse risks.

This diagnostic is intentionally read-only with respect to the model, data, and
checkpoint.  Its only write is the JSON path supplied with ``--output-json``.
Ground-truth PSTHs are loaded through ``Data.data_handler.load_gt_data`` so the
audit uses the same target selection and binning as production training.

Example
-------
python Archive/Debugging/audit_rate_collapse.py ^
    100_epoch_post_strf_fix_4.mat ^
    --config simulation_config.yaml ^
    --output-json Archive/Debugging/rate_collapse_audit_100_epoch_post_strf_fix_4.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data import data_handler  # noqa: E402  (repo path is inserted above)


# Bounds currently enforced by Simulation/update_handler.py.  These are kept
# here explicitly so the audit reports the behavior of the saved production
# model rather than guessing bounds from initialization ranges.
PARAMETER_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "strf_gain": (0.001, None),
    "strf_alpha": (0.1, 250.0),
    "output_ad": (0.0, None),
    "abs_ref": (0.0, None),
    "rel_ref_a": (0.001, None),
    "rel_ref_b": (0.0, None),
    "rel_ref_c": (0.2, 1.0),
    "on_ron_gsyn": (0.001, None),
    "off_ron_gsyn": (0.001, None),
    "on_sonoff_gsyn": (0.001, None),
    "off_sonoff_gsyn": (0.001, None),
    "sonoff_ron_gsyn": (0.001, None),
}


def finite_float(value: Any) -> float | None:
    """Return a JSON-safe float, using None for NaN and infinities."""
    result = float(value)
    return result if np.isfinite(result) else None


def json_array(value: np.ndarray) -> list[Any]:
    """Convert a numeric array to nested JSON-safe lists."""
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        return [finite_float(item) for item in arr.ravel().tolist()]
    return arr.tolist()


def numeric_summary(value: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(value, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": int(arr.size),
            "finite_count": 0,
            "min": None,
            "q01": None,
            "q10": None,
            "median": None,
            "mean": None,
            "q90": None,
            "q99": None,
            "max": None,
        }
    quantiles = np.quantile(finite, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": finite_float(np.min(finite)),
        "q01": finite_float(quantiles[0]),
        "q10": finite_float(quantiles[1]),
        "median": finite_float(quantiles[2]),
        "mean": finite_float(np.mean(finite)),
        "q90": finite_float(quantiles[3]),
        "q99": finite_float(quantiles[4]),
        "max": finite_float(np.max(finite)),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return finite_float(np.corrcoef(x, y)[0, 1])


def error_metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    difference = np.asarray(model, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "mae_hz": finite_float(np.mean(np.abs(difference))),
        "rmse_hz": finite_float(np.sqrt(np.mean(difference**2))),
        "mean_error_hz": finite_float(np.mean(difference)),
        "correlation": safe_correlation(model, reference),
    }


def extract_mat_struct(mat: dict[str, Any], key: str) -> dict[str, np.ndarray] | None:
    value = mat.get(key)
    if not isinstance(value, np.ndarray) or value.size == 0:
        return None
    obj = value.ravel()[0]
    fields = getattr(obj, "_fieldnames", None)
    if not fields:
        return None
    return {str(field).lower(): np.asarray(getattr(obj, field)) for field in fields}


def choose_axes(
    shape: tuple[int, ...],
    expected_cells: int,
    expected_trials: int,
    expected_batches: int | None,
    expected_timesteps: int,
) -> tuple[int | None, int, int, int]:
    """Identify batch, trial, cell, and time axes in a squeezed output array."""
    time_matches = [index for index, size in enumerate(shape) if size == expected_timesteps]
    time_axis = time_matches[-1] if time_matches else int(np.argmax(shape))

    remaining = [axis for axis in range(len(shape)) if axis != time_axis]
    cell_matches = [axis for axis in remaining if shape[axis] == expected_cells]
    if expected_cells == 1 and not cell_matches:
        cell_axis = -1  # singleton cell axis was squeezed out
    elif len(cell_matches) == 1:
        cell_axis = cell_matches[0]
        remaining.remove(cell_axis)
    else:
        raise ValueError(
            f"Could not uniquely identify cell axis of size {expected_cells} in output shape {shape}."
        )

    if cell_axis == -1:
        remaining = [axis for axis in range(len(shape)) if axis != time_axis]

    batch_axis: int | None = None
    if expected_batches is not None:
        batch_matches = [axis for axis in remaining if shape[axis] == expected_batches]
        if len(batch_matches) == 1:
            batch_axis = batch_matches[0]

    trial_candidates = [axis for axis in remaining if axis != batch_axis and shape[axis] == expected_trials]
    if len(trial_candidates) == 1:
        trial_axis = trial_candidates[0]
    elif len(trial_candidates) > 1:
        # Production order is batch, trial, cell, time.  Prefer the later axis.
        trial_axis = max(trial_candidates)
    elif len(remaining) == 1:
        trial_axis = remaining[0]
    else:
        raise ValueError(
            f"Could not identify trial axis of size {expected_trials} in output shape {shape}."
        )

    if batch_axis is None:
        leftovers = [axis for axis in remaining if axis != trial_axis]
        if len(leftovers) == 1:
            batch_axis = leftovers[0]
        elif len(leftovers) > 1:
            raise ValueError(f"Could not uniquely identify batch axis in output shape {shape}.")

    return batch_axis, trial_axis, cell_axis, time_axis


def normalize_output(
    raw_output: np.ndarray,
    expected_cells: int,
    expected_trials: int,
    expected_batches: int | None,
    expected_timesteps: int,
) -> np.ndarray:
    """Normalize common MATLAB output layouts to batch x trial x cell x time."""
    output = np.asarray(raw_output)
    while output.ndim > 4:
        singleton_axes = [axis for axis, size in enumerate(output.shape) if size == 1]
        if not singleton_axes:
            raise ValueError(f"Unsupported output shape with more than four non-singleton axes: {output.shape}")
        output = np.squeeze(output, axis=singleton_axes[0])

    output = np.squeeze(output)
    if output.ndim < 2 or output.ndim > 4:
        raise ValueError(f"Unsupported squeezed output shape: {output.shape}")

    batch_axis, trial_axis, cell_axis, time_axis = choose_axes(
        output.shape,
        expected_cells,
        expected_trials,
        expected_batches,
        expected_timesteps,
    )

    if cell_axis == -1:
        output = np.expand_dims(output, axis=output.ndim)
        cell_axis = output.ndim - 1

    if batch_axis is None:
        output = np.expand_dims(output, axis=output.ndim)
        batch_axis = output.ndim - 1

    normalized = np.transpose(output, (batch_axis, trial_axis, cell_axis, time_axis))
    if normalized.shape[2] != expected_cells:
        raise ValueError(
            f"Normalized output has {normalized.shape[2]} cells; expected {expected_cells}."
        )
    return np.asarray(normalized, dtype=np.float64)


def load_targets(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target_config = copy.deepcopy(config)
    target_config["simulation"]["device"] = "cpu"
    loaded = data_handler.load_gt_data(target_config)
    psth = loaded["psth_holder"].detach().cpu().numpy().astype(np.float64, copy=False)
    raster = loaded["raster_holder"].detach().cpu().numpy().astype(np.float64, copy=False)
    return psth, raster


def bound_count(values: np.ndarray, bound: float | None) -> int:
    if bound is None:
        return 0
    tolerance = max(1e-7, abs(bound) * 1e-5)
    return int(np.sum(np.isclose(values, bound, rtol=1e-5, atol=tolerance)))


def parameter_audit(
    trackers: dict[str, np.ndarray] | None,
    checkpoints: dict[str, np.ndarray] | None,
    slope_window: int,
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if trackers is None and checkpoints is None:
        warnings.append("MAT file has no readable named parameter structs; clamp and slope audit was skipped.")
        return {}, {}

    names = sorted(set((trackers or {}).keys()) | set((checkpoints or {}).keys()))
    report: dict[str, Any] = {}
    current_values: dict[str, np.ndarray] = {}

    for name in names:
        tracker = np.asarray((trackers or {}).get(name, np.array([])), dtype=np.float64)
        checkpoint = np.asarray((checkpoints or {}).get(name, np.array([])), dtype=np.float64)

        has_tracker_epochs = tracker.ndim >= 3
        if checkpoint.size:
            current = np.squeeze(checkpoint)
        elif has_tracker_epochs:
            current = np.asarray(tracker[..., -1])
        elif tracker.size:
            current = np.squeeze(tracker)
        else:
            continue

        current_values[name] = current
        entry: dict[str, Any] = {
            "current": numeric_summary(current),
            "tracker_shape": list(tracker.shape) if tracker.size else None,
            "checkpoint_shape": list(checkpoint.shape) if checkpoint.size else None,
        }

        if has_tracker_epochs:
            epochs = tracker.shape[-1]
            n_slope = min(max(2, slope_window), epochs)
            x = np.arange(n_slope, dtype=np.float64)
            centered = x - np.mean(x)
            denominator = np.sum(centered**2)
            candidate_slopes = np.sum(tracker[..., -n_slope:] * centered, axis=-1) / denominator
            population_means = np.mean(tracker, axis=tuple(range(tracker.ndim - 1)))
            entry.update(
                {
                    "epochs_tracked": int(epochs),
                    "initial_population_mean": finite_float(population_means[0]),
                    "last_tracker_population_mean": finite_float(population_means[-1]),
                    "slope_window_epochs": int(n_slope),
                    "population_mean_slope_per_epoch": finite_float(np.mean(candidate_slopes)),
                    "candidate_slope": numeric_summary(candidate_slopes),
                    "candidate_fraction_positive_slope": finite_float(np.mean(candidate_slopes > 0)),
                }
            )
            if checkpoint.size and checkpoint.shape == tracker[..., -1].shape:
                final_step = checkpoint - tracker[..., -1]
                entry["last_update"] = {
                    "signed": numeric_summary(final_step),
                    "absolute": numeric_summary(np.abs(final_step)),
                }

        lower, upper = PARAMETER_BOUNDS.get(name, (None, None))
        lower_count = bound_count(current, lower)
        upper_count = bound_count(current, upper)
        entry["enforced_bounds"] = {"lower": lower, "upper": upper}
        entry["at_lower_bound_count"] = lower_count
        entry["at_upper_bound_count"] = upper_count
        entry["at_lower_bound_fraction"] = finite_float(lower_count / current.size)
        entry["at_upper_bound_fraction"] = finite_float(upper_count / current.size)

        if lower_count / current.size >= 0.10:
            warnings.append(
                f"{name}: {100 * lower_count / current.size:.1f}% of candidates are at the lower clamp."
            )
        if upper_count / current.size >= 0.10:
            warnings.append(
                f"{name}: {100 * upper_count / current.size:.1f}% of candidates are at the upper clamp."
            )
        report[name] = entry

    return report, current_values


def candidate_index_records(
    mask: np.ndarray,
    cell_targets: list[int],
    limit: int = 200,
) -> tuple[list[dict[str, int]], bool]:
    indices = np.argwhere(mask)
    records = [
        {
            "batch_index_zero_based": int(batch),
            "batch_index_one_based": int(batch + 1),
            "cell_axis_zero_based": int(cell),
            "cell_id": int(cell_targets[cell]),
        }
        for batch, cell in indices[:limit]
    ]
    return records, bool(len(indices) > limit)


def analyze(
    mat_path: Path,
    config_path: Path,
    early_bins_requested: int,
    slope_window: int,
) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    simulation = config["simulation"]
    cell_targets = [int(cell) for cell in simulation["cell_targets"]]
    expected_cells = len(cell_targets)
    dt_ms = float(simulation["dt"])
    granularity = int(simulation["PSTH_granularity"])
    expected_timesteps = int(simulation["sim_len"])
    expected_batches = int(simulation["batch_size"])

    target_psth, target_raster = load_targets(config)
    target_trials = int(target_raster.shape[1])

    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    if "output" not in mat:
        raise KeyError(f"{mat_path} has no 'output' variable.")
    output = normalize_output(
        mat["output"],
        expected_cells=expected_cells,
        expected_trials=target_trials,
        expected_batches=expected_batches,
        expected_timesteps=expected_timesteps,
    )
    batches, model_trials, output_cells, output_timesteps = output.shape

    warnings: list[str] = []
    if model_trials != target_trials:
        warnings.append(
            f"Model has {model_trials} trials but production target has {target_trials}; raw count loss is not trial-normalized."
        )
    if output_cells != expected_cells:
        warnings.append(f"Output has {output_cells} cells but config requests {expected_cells}.")
    if output_timesteps != expected_timesteps:
        warnings.append(
            f"Output has {output_timesteps} timesteps but config requests {expected_timesteps}."
        )

    complete_model_bins = output_timesteps // granularity
    common_bins = min(complete_model_bins, target_psth.shape[-1])
    if common_bins <= 0:
        raise ValueError("No complete common PSTH bins are available.")
    if complete_model_bins != target_psth.shape[-1]:
        warnings.append(
            f"Model has {complete_model_bins} complete bins and target has {target_psth.shape[-1]}; using {common_bins}."
        )

    window_steps = common_bins * granularity
    sim_psth = output[..., :window_steps].reshape(
        batches, model_trials, output_cells, common_bins, granularity
    ).sum(axis=(1, 4))
    target_psth = target_psth[:, :common_bins]

    bin_seconds = granularity * dt_ms / 1000.0
    window_seconds = common_bins * bin_seconds
    full_seconds = output_timesteps * dt_ms / 1000.0

    model_candidate_rate_window = sim_psth.sum(axis=-1) / (model_trials * window_seconds)
    model_candidate_rate_full = output.sum(axis=(1, 3)) / (model_trials * full_seconds)
    model_cell_rate = np.mean(model_candidate_rate_window, axis=0)
    target_cell_rate = target_psth.sum(axis=-1) / (target_trials * window_seconds)

    # Two useful limits of the 0.5-count Poisson MSE bias.  A model that can
    # fit every bin independently approaches the binwise-clipped prediction;
    # a broad/constant template per cell approaches the constant-rate one.
    poisson_binwise_count = np.maximum(target_psth - 0.5, 0.0)
    poisson_binwise_cell_rate = poisson_binwise_count.sum(axis=-1) / (
        target_trials * window_seconds
    )
    target_mean_count_per_bin = np.mean(target_psth, axis=-1)
    poisson_constant_count = np.maximum(target_mean_count_per_bin - 0.5, 0.0)
    poisson_constant_cell_rate = poisson_constant_count / (target_trials * bin_seconds)
    bias_hz = 0.5 / (target_trials * bin_seconds)

    rate_ratio = np.divide(
        model_cell_rate,
        target_cell_rate,
        out=np.full_like(model_cell_rate, np.nan),
        where=target_cell_rate > 0,
    )
    model_vs_target = error_metrics(model_cell_rate, target_cell_rate)
    model_vs_constant_bias = error_metrics(model_cell_rate, poisson_constant_cell_rate)
    model_vs_binwise_bias = error_metrics(model_cell_rate, poisson_binwise_cell_rate)

    residual = sim_psth - target_psth[None, :, :]
    actual_sse = float(np.mean(residual**2, axis=(0, 1)).sum())
    silent_sse = float(np.mean(target_psth**2, axis=0).sum())

    numerator = float(np.sum(sim_psth * target_psth[None, :, :]))
    denominator = float(np.sum(sim_psth**2))
    unconstrained_scale = numerator / denominator if denominator > 0 else np.nan
    nonnegative_scale = max(0.0, unconstrained_scale) if np.isfinite(unconstrained_scale) else 0.0
    scaled_residual = nonnegative_scale * sim_psth - target_psth[None, :, :]
    scaled_sse = float(np.mean(scaled_residual**2, axis=(0, 1)).sum())
    count_matching_scale = (
        float(batches * np.sum(target_psth) / np.sum(sim_psth)) if np.sum(sim_psth) > 0 else np.nan
    )

    per_cell_numerator = np.sum(sim_psth * target_psth[None, :, :], axis=(0, 2))
    per_cell_denominator = np.sum(sim_psth**2, axis=(0, 2))
    per_cell_scale = np.divide(
        per_cell_numerator,
        per_cell_denominator,
        out=np.full(expected_cells, np.nan),
        where=per_cell_denominator > 0,
    )
    per_cell_scale = np.maximum(per_cell_scale, 0.0)

    early_bins = min(max(1, int(early_bins_requested)), common_bins)
    target_total = float(np.sum(target_psth))
    model_total = float(np.sum(sim_psth))
    target_early_fraction = float(np.sum(target_psth[:, :early_bins]) / target_total) if target_total else np.nan
    model_early_fraction = float(np.sum(sim_psth[..., :early_bins]) / model_total) if model_total else np.nan
    target_cell_early_fraction = np.divide(
        np.sum(target_psth[:, :early_bins], axis=-1),
        np.sum(target_psth, axis=-1),
        out=np.full(expected_cells, np.nan),
        where=np.sum(target_psth, axis=-1) > 0,
    )
    model_cell_psth = np.mean(sim_psth, axis=0)
    model_cell_early_fraction = np.divide(
        np.sum(model_cell_psth[:, :early_bins], axis=-1),
        np.sum(model_cell_psth, axis=-1),
        out=np.full(expected_cells, np.nan),
        where=np.sum(model_cell_psth, axis=-1) > 0,
    )

    silent_mask = output.sum(axis=(1, 3)) == 0
    near_silent_mask = model_candidate_rate_full < 1.0
    silent_records, silent_truncated = candidate_index_records(silent_mask, cell_targets)
    near_silent_records, near_silent_truncated = candidate_index_records(near_silent_mask, cell_targets)

    trackers = extract_mat_struct(mat, "params")
    checkpoints = extract_mat_struct(mat, "checkpoint_params")
    if trackers is not None and checkpoints is None:
        warnings.append(
            "MAT file has trackers but no checkpoint_params; current parameters are approximated by the final tracker."
        )
    parameters, current_params = parameter_audit(trackers, checkpoints, slope_window, warnings)

    startup_risk: dict[str, Any] = {}
    abs_ref = current_params.get("abs_ref")
    if abs_ref is not None:
        fake_spike_age_ms = 30 * dt_ms
        startup_delay_ms = np.maximum(abs_ref - fake_spike_age_ms, 0.0)
        startup_risk["fake_spike_age_ms"] = finite_float(fake_spike_age_ms)
        startup_risk["abs_ref_greater_than_fake_spike_age_count"] = int(np.sum(abs_ref > fake_spike_age_ms))
        startup_risk["abs_ref_greater_than_fake_spike_age_fraction"] = finite_float(
            np.mean(abs_ref > fake_spike_age_ms)
        )
        startup_risk["forced_startup_refractory_delay_ms"] = numeric_summary(startup_delay_ms)
        if np.max(abs_ref) > 50:
            warnings.append(f"abs_ref reaches {np.max(abs_ref):.1f} ms and has no configured upper clamp.")

    rel_a = current_params.get("rel_ref_a")
    rel_b = current_params.get("rel_ref_b")
    if rel_a is not None and rel_b is not None and rel_a.shape == rel_b.shape:
        midpoint_steps = np.divide(rel_b, rel_a, out=np.full_like(rel_b, np.inf), where=rel_a > 0)
        extra_midpoint_ms = np.maximum(midpoint_steps - 30.0, 0.0) * dt_ms
        startup_risk["relative_recovery_midpoint_after_t0_ms"] = numeric_summary(extra_midpoint_ms)
        startup_risk["relative_recovery_midpoint_over_one_second_count"] = int(
            np.sum(extra_midpoint_ms > 1000.0)
        )

    if finite_float(np.mean(model_cell_rate)) is not None and np.mean(model_cell_rate) < 0.8 * np.mean(target_cell_rate):
        warnings.append(
            f"Population rate is {100 * np.mean(model_cell_rate) / np.mean(target_cell_rate):.1f}% of target."
        )
    if (
        model_vs_target["rmse_hz"] is not None
        and model_vs_constant_bias["rmse_hz"] is not None
        and model_vs_constant_bias["rmse_hz"] < model_vs_target["rmse_hz"]
    ):
        warnings.append(
            "Model cell rates are closer in RMSE to the constant-template 0.5-count bias prediction than to target rates."
        )
    if np.any(silent_mask):
        warnings.append(f"{int(np.sum(silent_mask))} batch/cell candidates are completely silent.")
    if np.any(near_silent_mask):
        warnings.append(f"{int(np.sum(near_silent_mask))} batch/cell candidates fire below 1 Hz.")
    if silent_sse <= actual_sse:
        warnings.append("All-silent output has SSE no worse than the saved output under the production loss.")
    if np.isfinite(model_early_fraction) and np.isfinite(target_early_fraction) and target_early_fraction > 0:
        early_ratio = model_early_fraction / target_early_fraction
        if early_ratio < 0.8 or early_ratio > 1.2:
            warnings.append(
                f"First {early_bins} bins contain {100 * early_ratio:.1f}% of the target-normalized early spike fraction."
            )

    adam_beta1 = config.get("Adam", {}).get("beta1")
    if adam_beta1 is not None and float(adam_beta1) < 0.1:
        warnings.append(f"Adam beta1={float(adam_beta1):g} provides almost no epoch-to-epoch gradient smoothing.")

    saved_last_loss = None
    if "losses" in mat and np.asarray(mat["losses"]).size:
        saved_last_loss = finite_float(np.asarray(mat["losses"], dtype=np.float64).ravel()[-1])
        if saved_last_loss is not None:
            relative_difference = abs(saved_last_loss - actual_sse) / max(1.0, abs(saved_last_loss))
            if relative_difference > 1e-3:
                warnings.append(
                    f"Recomputed SSE ({actual_sse:.6g}) differs from saved last loss ({saved_last_loss:.6g})."
                )

    if trackers is not None and checkpoints is not None:
        warnings.append(
            "Saved output is generated before the final Adam update, while checkpoint_params stores the post-update state."
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "mat_path": str(mat_path),
            "config_path": str(config_path),
            "production_target_loader": "Data.data_handler.load_gt_data",
        },
        "dimensions": {
            "batches": int(batches),
            "model_trials": int(model_trials),
            "target_trials": int(target_trials),
            "cells": int(output_cells),
            "timesteps": int(output_timesteps),
            "dt_ms": dt_ms,
            "psth_granularity_steps": int(granularity),
            "bin_ms": finite_float(bin_seconds * 1000.0),
            "common_complete_bins": int(common_bins),
            "training_window_seconds": finite_float(window_seconds),
            "full_output_seconds": finite_float(full_seconds),
        },
        "rates": {
            "target_population_mean_hz": finite_float(np.mean(target_cell_rate)),
            "target_cell_hz": json_array(target_cell_rate),
            "target_cell_hz_summary": numeric_summary(target_cell_rate),
            "model_population_mean_hz_training_window": finite_float(np.mean(model_candidate_rate_window)),
            "model_population_mean_hz_full_output": finite_float(np.mean(model_candidate_rate_full)),
            "model_candidate_hz_summary": numeric_summary(model_candidate_rate_full),
            "model_cell_mean_hz": json_array(model_cell_rate),
            "model_cell_mean_hz_summary": numeric_summary(model_cell_rate),
            "model_to_target_cell_ratio_summary": numeric_summary(rate_ratio),
            "poisson_half_count_bias_hz": finite_float(bias_hz),
            "poisson_binwise_bias_predicted_cell_hz": json_array(poisson_binwise_cell_rate),
            "poisson_binwise_bias_predicted_population_mean_hz": finite_float(np.mean(poisson_binwise_cell_rate)),
            "poisson_constant_template_bias_predicted_cell_hz": json_array(poisson_constant_cell_rate),
            "poisson_constant_template_bias_predicted_population_mean_hz": finite_float(np.mean(poisson_constant_cell_rate)),
            "model_vs_target": model_vs_target,
            "model_vs_poisson_binwise_bias_prediction": model_vs_binwise_bias,
            "model_vs_poisson_constant_template_bias_prediction": model_vs_constant_bias,
        },
        "psth_sse": {
            "production_definition_recomputed": finite_float(actual_sse),
            "saved_last_loss": saved_last_loss,
            "all_silent": finite_float(silent_sse),
            "all_silent_to_actual_ratio": finite_float(silent_sse / actual_sse) if actual_sse else None,
            "actual_improvement_over_all_silent": finite_float(silent_sse - actual_sse),
        },
        "optimal_psth_scaling": {
            "global_unconstrained_least_squares_scale": finite_float(unconstrained_scale),
            "global_nonnegative_least_squares_scale": finite_float(nonnegative_scale),
            "global_scaled_sse": finite_float(scaled_sse),
            "global_scaled_sse_improvement": finite_float(actual_sse - scaled_sse),
            "total_count_matching_scale": finite_float(count_matching_scale),
            "per_cell_nonnegative_scale": json_array(per_cell_scale),
            "per_cell_nonnegative_scale_summary": numeric_summary(per_cell_scale),
        },
        "early_bin_fraction": {
            "early_bins": int(early_bins),
            "early_window_ms": finite_float(early_bins * bin_seconds * 1000.0),
            "target_global_fraction": finite_float(target_early_fraction),
            "model_global_fraction": finite_float(model_early_fraction),
            "model_to_target_fraction_ratio": finite_float(model_early_fraction / target_early_fraction)
            if target_early_fraction
            else None,
            "target_cell_fraction_summary": numeric_summary(target_cell_early_fraction),
            "model_cell_fraction_summary": numeric_summary(model_cell_early_fraction),
        },
        "silent_candidates": {
            "total_candidates": int(batches * output_cells),
            "silent_count": int(np.sum(silent_mask)),
            "silent_fraction": finite_float(np.mean(silent_mask)),
            "silent_indices": silent_records,
            "silent_indices_truncated": silent_truncated,
            "below_1_hz_count": int(np.sum(near_silent_mask)),
            "below_1_hz_fraction": finite_float(np.mean(near_silent_mask)),
            "below_1_hz_indices": near_silent_records,
            "below_1_hz_indices_truncated": near_silent_truncated,
        },
        "parameter_audit": parameters,
        "startup_refractory_risk": startup_risk,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", type=Path, help="Saved simulation/checkpoint MAT file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "simulation_config.yaml",
        help="Production YAML configuration (default: repository simulation_config.yaml).",
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Destination JSON report.")
    parser.add_argument(
        "--early-bins",
        type=int,
        default=10,
        help="Number of initial PSTH bins used for the early-spike fraction (default: 10).",
    )
    parser.add_argument(
        "--slope-window",
        type=int,
        default=10,
        help="Final tracker epochs used for parameter slopes (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mat_path = args.mat_path.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()

    report = analyze(
        mat_path=mat_path,
        config_path=config_path,
        early_bins_requested=args.early_bins,
        slope_window=args.slope_window,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")

    rates = report["rates"]
    silent = report["silent_candidates"]
    print(f"Wrote rate-collapse audit: {output_path}")
    print(
        "Target/model/binwise-bias/constant-template-bias population rates: "
        f"{rates['target_population_mean_hz']:.3f} / "
        f"{rates['model_population_mean_hz_full_output']:.3f} / "
        f"{rates['poisson_binwise_bias_predicted_population_mean_hz']:.3f} / "
        f"{rates['poisson_constant_template_bias_predicted_population_mean_hz']:.3f} Hz"
    )
    print(
        f"Silent candidates: {silent['silent_count']}/{silent['total_candidates']}; "
        f"warnings: {len(report['warnings'])}"
    )


if __name__ == "__main__":
    main()
