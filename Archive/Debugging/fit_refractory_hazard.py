"""Fit an offline refractory hazard model to production ACX spike targets.

The script does not modify simulation code, data, or checkpoints.  It loads
target rasters through ``Data.data_handler.load_gt_data`` and writes one JSON
report.  Absolute refractory duration is profiled as a discrete parameter;
the smooth parameters are estimated by a full Bernoulli event/no-event
likelihood, including no-spike bins after the final event in each trial.

The fitted conditional intensity is

    lambda(t, age) = exp(B(t) @ beta) * recovery(age)

with a regularized low-dimensional piecewise-linear basis B(t), a hard
absolute refractory interval, and a normalized sigmoid recovery curve.  The
normalization fixes recovery(infinity)=1, avoiding confounding between the
baseline intensity and the asymptotic recovery probability.

Example
-------
python Archive/Debugging/fit_refractory_hazard.py ^
    --config simulation_config.yaml --cells 7 ^
    --abs-ref-grid-ms 0:2:0.1 --held-out-trial 10 ^
    --output-json Archive/Debugging/refractory_hazard_fit_cell7.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import expit


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data import data_handler  # noqa: E402


@dataclass(frozen=True)
class FitContext:
    basis: np.ndarray
    second_difference: np.ndarray
    spikes: np.ndarray
    ages_ms: np.ndarray
    abs_ref_ms: float
    dt_seconds: float
    smoothness: float


def finite_float(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def numeric_summary(value: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(value, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": finite_float(np.min(finite)),
        "q10": finite_float(np.quantile(finite, 0.10)),
        "median": finite_float(np.median(finite)),
        "mean": finite_float(np.mean(finite)),
        "q90": finite_float(np.quantile(finite, 0.90)),
        "max": finite_float(np.max(finite)),
    }


def parse_cells(value: str) -> list[int]:
    cells: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            step = 1 if stop >= start else -1
            cells.extend(range(start, stop + step, step))
        else:
            cells.append(int(token))
    unique = list(dict.fromkeys(cells))
    if not unique:
        raise ValueError("At least one cell must be selected.")
    return unique


def parse_grid(value: str) -> np.ndarray:
    text = value.strip()
    if ":" in text:
        pieces = [float(piece) for piece in text.split(":")]
        if len(pieces) != 3:
            raise ValueError("Grid range must use start:stop:step.")
        start, stop, step = pieces
        if step <= 0 or stop < start:
            raise ValueError("Grid requires positive step and stop >= start.")
        values = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    else:
        values = np.array([float(piece) for piece in text.split(",") if piece.strip()])
    if not values.size or np.any(values < 0):
        raise ValueError("Absolute refractory grid must contain nonnegative values.")
    return values


def snap_grid_to_dt(values_ms: np.ndarray, dt_ms: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    snapped = np.round(np.asarray(values_ms) / dt_ms) * dt_ms
    changes = [
        {"requested_ms": finite_float(requested), "snapped_ms": finite_float(actual)}
        for requested, actual in zip(values_ms, snapped)
        if not np.isclose(requested, actual, rtol=0, atol=1e-10)
    ]
    return np.unique(np.round(snapped, decimals=12)), changes


def piecewise_linear_basis(timesteps: int, basis_count: int) -> tuple[np.ndarray, np.ndarray]:
    if basis_count < 3:
        raise ValueError("At least three baseline basis functions are required.")
    coordinate = np.linspace(0.0, basis_count - 1, timesteps)
    lower = np.floor(coordinate).astype(np.int64)
    lower = np.minimum(lower, basis_count - 2)
    fraction = coordinate - lower
    basis = np.zeros((timesteps, basis_count), dtype=np.float64)
    rows = np.arange(timesteps)
    basis[rows, lower] = 1.0 - fraction
    basis[rows, lower + 1] += fraction

    second_difference = np.zeros((basis_count - 2, basis_count), dtype=np.float64)
    rows = np.arange(basis_count - 2)
    second_difference[rows, rows] = 1.0
    second_difference[rows, rows + 1] = -2.0
    second_difference[rows, rows + 2] = 1.0
    return basis, second_difference


def ages_before_bin(spikes: np.ndarray, dt_ms: float) -> np.ndarray:
    """Return age since the previous spike before each bin; trial starts are recovered."""
    spikes = np.asarray(spikes, dtype=bool)
    ages = np.full(spikes.shape, np.inf, dtype=np.float64)
    for trial_index in range(spikes.shape[0]):
        last_spike: int | None = None
        for timestep in range(spikes.shape[1]):
            if last_spike is not None:
                ages[trial_index, timestep] = (timestep - last_spike) * dt_ms
            if spikes[trial_index, timestep]:
                last_spike = timestep
    return ages


def recovery_and_derivatives(
    ages_ms: np.ndarray,
    abs_ref_ms: float,
    log_lag_ms: float,
    log_width_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    lag_ms = math.exp(log_lag_ms)
    width_ms = math.exp(log_width_ms)
    tau50_ms = abs_ref_ms + lag_ms

    recovery = np.zeros_like(ages_ms, dtype=np.float64)
    derivative_log_lag = np.zeros_like(ages_ms, dtype=np.float64)
    derivative_log_width = np.zeros_like(ages_ms, dtype=np.float64)

    no_previous_spike = np.isinf(ages_ms)
    recovery[no_previous_spike] = 1.0
    smooth = np.isfinite(ages_ms) & (ages_ms > abs_ref_ms)
    z = (ages_ms[smooth] - tau50_ms) / width_ms
    value = expit(z)
    logistic_derivative = value * (1.0 - value)
    recovery[smooth] = value
    derivative_log_lag[smooth] = -logistic_derivative * lag_ms / width_ms
    derivative_log_width[smooth] = -logistic_derivative * z
    return recovery, derivative_log_lag, derivative_log_width, tau50_ms, width_ms


def hazard_forward(
    parameters: np.ndarray,
    context: FitContext,
) -> dict[str, Any]:
    basis_count = context.basis.shape[1]
    beta = parameters[:basis_count]
    log_lag_ms, log_width_ms = parameters[-2:]
    eta = context.basis @ beta
    baseline_hz = np.exp(eta)
    base_exposure = context.dt_seconds * baseline_hz
    recovery, derivative_log_lag, derivative_log_width, tau50_ms, width_ms = (
        recovery_and_derivatives(
            context.ages_ms,
            context.abs_ref_ms,
            log_lag_ms,
            log_width_ms,
        )
    )
    cumulative_hazard_bin = base_exposure[None, :] * recovery
    return {
        "beta": beta,
        "baseline_hz": baseline_hz,
        "base_exposure": base_exposure,
        "recovery": recovery,
        "derivative_log_lag": derivative_log_lag,
        "derivative_log_width": derivative_log_width,
        "x": cumulative_hazard_bin,
        "tau50_ms": tau50_ms,
        "width_ms": width_ms,
    }


def unpenalized_nll(spikes: np.ndarray, x: np.ndarray) -> float:
    event_x = x[spikes]
    if event_x.size and np.any(event_x <= 0):
        return math.inf
    event_term = -np.sum(np.log(-np.expm1(-event_x))) if event_x.size else 0.0
    no_event_term = np.sum(x) - np.sum(event_x)
    return float(event_term + no_event_term)


def hazard_value_and_gradient(
    parameters: np.ndarray,
    context: FitContext,
) -> tuple[float, np.ndarray]:
    forward = hazard_forward(parameters, context)
    x = forward["x"]
    spikes = context.spikes
    event_x = x[spikes]
    if event_x.size and np.any(event_x <= 0):
        return 1e100, np.zeros_like(parameters)

    nll = unpenalized_nll(spikes, x)
    curvature = context.second_difference @ forward["beta"]
    penalty = 0.5 * context.smoothness * float(curvature @ curvature)

    derivative_x = np.ones_like(x)
    if event_x.size:
        derivative_x[spikes] = -1.0 / np.expm1(event_x)

    derivative_eta = np.sum(derivative_x * x, axis=0)
    gradient_beta = context.basis.T @ derivative_eta
    gradient_beta += context.smoothness * context.second_difference.T @ curvature

    derivative_recovery = derivative_x * forward["base_exposure"][None, :]
    gradient_log_lag = float(
        np.sum(derivative_recovery * forward["derivative_log_lag"])
    )
    gradient_log_width = float(
        np.sum(derivative_recovery * forward["derivative_log_width"])
    )
    gradient = np.concatenate(
        [gradient_beta, np.array([gradient_log_lag, gradient_log_width])]
    )
    return float(nll + penalty), gradient


def baseline_value_and_gradient(
    beta: np.ndarray,
    basis: np.ndarray,
    second_difference: np.ndarray,
    spikes: np.ndarray,
    dt_seconds: float,
    smoothness: float,
) -> tuple[float, np.ndarray]:
    eta = basis @ beta
    x = dt_seconds * np.exp(eta)
    events_by_time = np.sum(spikes, axis=0).astype(np.float64)
    trials = spikes.shape[0]
    event_term = -np.sum(events_by_time * np.log(-np.expm1(-x)))
    no_event_term = np.sum((trials - events_by_time) * x)
    curvature = second_difference @ beta
    penalty = 0.5 * smoothness * float(curvature @ curvature)

    derivative_x = (trials - events_by_time) - events_by_time / np.expm1(x)
    derivative_eta = derivative_x * x
    gradient = basis.T @ derivative_eta
    gradient += smoothness * second_difference.T @ curvature
    return float(event_term + no_event_term + penalty), gradient


def model_metrics(parameters: np.ndarray, context: FitContext) -> dict[str, Any]:
    forward = hazard_forward(parameters, context)
    x = forward["x"]
    nll = unpenalized_nll(context.spikes, x)
    probability = -np.expm1(-x)
    duration_seconds = context.spikes.shape[1] * context.dt_seconds
    observed_spikes = int(np.sum(context.spikes))
    expected_spikes = float(np.sum(probability))
    return {
        "trials": int(context.spikes.shape[0]),
        "bins_per_trial": int(context.spikes.shape[1]),
        "observed_spikes": observed_spikes,
        "expected_spikes": finite_float(expected_spikes),
        "observed_rate_hz": finite_float(
            observed_spikes / (context.spikes.shape[0] * duration_seconds)
        ),
        "expected_rate_hz": finite_float(
            expected_spikes / (context.spikes.shape[0] * duration_seconds)
        ),
        "negative_log_likelihood": finite_float(nll),
        "nll_per_bin": finite_float(nll / context.spikes.size),
        "mean_event_probability": finite_float(np.mean(probability)),
    }


def baseline_metrics(
    beta: np.ndarray,
    basis: np.ndarray,
    spikes: np.ndarray,
    dt_seconds: float,
) -> dict[str, Any]:
    x = dt_seconds * np.exp(basis @ beta)
    probability = -np.expm1(-x)
    events_by_time = np.sum(spikes, axis=0).astype(np.float64)
    trials = spikes.shape[0]
    nll = -np.sum(events_by_time * np.log(probability))
    nll += np.sum((trials - events_by_time) * x)
    duration_seconds = spikes.shape[1] * dt_seconds
    expected_spikes = float(trials * np.sum(probability))
    observed_spikes = int(np.sum(spikes))
    return {
        "trials": int(trials),
        "observed_spikes": observed_spikes,
        "expected_spikes": finite_float(expected_spikes),
        "observed_rate_hz": finite_float(observed_spikes / (trials * duration_seconds)),
        "expected_rate_hz": finite_float(expected_spikes / (trials * duration_seconds)),
        "negative_log_likelihood": finite_float(nll),
        "nll_per_bin": finite_float(nll / spikes.size),
    }


def current_model_mapping(abs_ref_ms: float, tau50_ms: float, width_ms: float, dt_ms: float) -> dict[str, float]:
    return {
        "abs_ref": finite_float(abs_ref_ms),
        "rel_ref_a": finite_float(dt_ms / (2.0 * width_ms)),
        "rel_ref_b": finite_float(tau50_ms / (2.0 * width_ms)),
        "rel_ref_c": 0.5,
    }


def fit_baseline(
    basis: np.ndarray,
    second_difference: np.ndarray,
    spikes: np.ndarray,
    dt_seconds: float,
    smoothness: float,
    maximum_iterations: int,
    beta_bounds: tuple[float, float],
) -> Any:
    duration_seconds = spikes.shape[1] * dt_seconds
    rate_hz = max(1e-3, np.sum(spikes) / (spikes.shape[0] * duration_seconds))
    initial = np.full(basis.shape[1], math.log(rate_hz), dtype=np.float64)
    return minimize(
        baseline_value_and_gradient,
        initial,
        args=(basis, second_difference, spikes, dt_seconds, smoothness),
        method="L-BFGS-B",
        jac=True,
        bounds=[beta_bounds] * basis.shape[1],
        options={"maxiter": maximum_iterations, "ftol": 1e-10, "gtol": 1e-6},
    )


def fit_profile_value(
    context: FitContext,
    baseline_beta: np.ndarray,
    initial_lag_ms: float,
    initial_width_ms: float,
    maximum_iterations: int,
    beta_bounds: tuple[float, float],
    recovery_bounds_ms: tuple[float, float],
) -> Any:
    initial = np.concatenate(
        [
            baseline_beta,
            np.log([initial_lag_ms, initial_width_ms]),
        ]
    )
    log_bounds = (math.log(recovery_bounds_ms[0]), math.log(recovery_bounds_ms[1]))
    bounds = [beta_bounds] * context.basis.shape[1] + [log_bounds, log_bounds]
    return minimize(
        hazard_value_and_gradient,
        initial,
        args=(context,),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maximum_iterations, "ftol": 1e-10, "gtol": 1e-6},
    )


def finite_difference_self_check() -> dict[str, Any]:
    timesteps = 60
    basis, second_difference = piecewise_linear_basis(timesteps, 6)
    spikes = np.zeros((2, timesteps), dtype=bool)
    spikes[0, [4, 22, 47]] = True
    spikes[1, [8, 31, 55]] = True
    dt_ms = 0.2
    context = FitContext(
        basis=basis,
        second_difference=second_difference,
        spikes=spikes,
        ages_ms=ages_before_bin(spikes, dt_ms),
        abs_ref_ms=1.0,
        dt_seconds=dt_ms / 1000.0,
        smoothness=0.7,
    )
    parameters = np.array(
        [2.5, 2.7, 2.9, 2.8, 2.6, 2.4, math.log(2.2), math.log(1.4)],
        dtype=np.float64,
    )
    value, analytic = hazard_value_and_gradient(parameters, context)
    step = 1e-6
    finite_difference = np.zeros_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = hazard_value_and_gradient(plus, context)[0]
        minus_value = hazard_value_and_gradient(minus, context)[0]
        finite_difference[index] = (plus_value - minus_value) / (2.0 * step)
    absolute_error = np.abs(analytic - finite_difference)
    relative_error = absolute_error / np.maximum(
        1.0, np.maximum(np.abs(analytic), np.abs(finite_difference))
    )
    passed = bool(np.max(relative_error) < 1e-5)
    return {
        "passed": passed,
        "objective": finite_float(value),
        "step": step,
        "analytic_gradient": analytic.tolist(),
        "finite_difference_gradient": finite_difference.tolist(),
        "maximum_absolute_error": finite_float(np.max(absolute_error)),
        "maximum_scaled_relative_error": finite_float(np.max(relative_error)),
        "tolerance": 1e-5,
    }


def fit_cell(
    cell_id: int,
    spikes: np.ndarray,
    dt_ms: float,
    abs_ref_grid_ms: np.ndarray,
    held_out_trial_one_based: int,
    basis_count: int,
    smoothness: float,
    maximum_iterations: int,
    minimum_recovery_ms: float,
    maximum_recovery_ms: float,
) -> dict[str, Any]:
    trials, timesteps = spikes.shape
    held_out_index = held_out_trial_one_based - 1
    if held_out_trial_one_based == 0:
        train_indices = np.arange(trials)
        held_out_indices = np.array([], dtype=np.int64)
    else:
        if held_out_index < 0 or held_out_index >= trials:
            raise ValueError(
                f"Held-out trial {held_out_trial_one_based} is invalid for {trials} trials."
            )
        train_indices = np.array([index for index in range(trials) if index != held_out_index])
        held_out_indices = np.array([held_out_index])

    train_spikes = spikes[train_indices]
    held_out_spikes = spikes[held_out_indices] if held_out_indices.size else None
    train_ages = ages_before_bin(train_spikes, dt_ms)
    held_out_ages = ages_before_bin(held_out_spikes, dt_ms) if held_out_spikes is not None else None
    basis, second_difference = piecewise_linear_basis(timesteps, basis_count)
    dt_seconds = dt_ms / 1000.0

    beta_bounds = (math.log(1e-3), math.log(1e4))
    baseline_result = fit_baseline(
        basis,
        second_difference,
        train_spikes,
        dt_seconds,
        smoothness,
        maximum_iterations,
        beta_bounds,
    )
    baseline_train = baseline_metrics(baseline_result.x, basis, train_spikes, dt_seconds)
    baseline_held_out = (
        baseline_metrics(baseline_result.x, basis, held_out_spikes, dt_seconds)
        if held_out_spikes is not None
        else None
    )

    finite_event_ages = train_ages[train_spikes & np.isfinite(train_ages)]
    minimum_train_isi_ms = float(np.min(finite_event_ages)) if finite_event_ages.size else math.inf
    profile: list[dict[str, Any]] = []
    successful: list[tuple[float, int, Any, FitContext]] = []

    for abs_ref_ms in abs_ref_grid_ms:
        if np.any(train_spikes & np.isfinite(train_ages) & (train_ages <= abs_ref_ms + 1e-12)):
            profile.append(
                {
                    "abs_ref_ms": finite_float(abs_ref_ms),
                    "valid": False,
                    "reason": (
                        "A training spike occurs at or before the proposed hard absolute refractory duration."
                    ),
                }
            )
            continue

        context = FitContext(
            basis=basis,
            second_difference=second_difference,
            spikes=train_spikes,
            ages_ms=train_ages,
            abs_ref_ms=float(abs_ref_ms),
            dt_seconds=dt_seconds,
            smoothness=smoothness,
        )
        result = fit_profile_value(
            context,
            baseline_result.x,
            initial_lag_ms=max(2.0, dt_ms),
            initial_width_ms=max(2.0, dt_ms),
            maximum_iterations=maximum_iterations,
            beta_bounds=beta_bounds,
            recovery_bounds_ms=(minimum_recovery_ms, maximum_recovery_ms),
        )
        forward = hazard_forward(result.x, context)
        train_metrics = model_metrics(result.x, context)
        held_out_metrics = None
        if held_out_spikes is not None and held_out_ages is not None:
            held_context = FitContext(
                basis=basis,
                second_difference=second_difference,
                spikes=held_out_spikes,
                ages_ms=held_out_ages,
                abs_ref_ms=float(abs_ref_ms),
                dt_seconds=dt_seconds,
                smoothness=0.0,
            )
            held_out_metrics = model_metrics(result.x, held_context)

        entry = {
            "abs_ref_ms": finite_float(abs_ref_ms),
            "valid": True,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "penalized_training_objective": finite_float(result.fun),
            "tau50_ms": finite_float(forward["tau50_ms"]),
            "width_ms": finite_float(forward["width_ms"]),
            "current_model_mapping": current_model_mapping(
                float(abs_ref_ms), forward["tau50_ms"], forward["width_ms"], dt_ms
            ),
            "training": train_metrics,
            "held_out": held_out_metrics,
        }
        profile_index = len(profile)
        profile.append(entry)
        if np.isfinite(result.fun):
            successful.append((float(result.fun), profile_index, result, context))

    if not successful:
        raise RuntimeError(f"No valid profile value could be fitted for cell {cell_id}.")
    _, selected_profile_index, selected_result, selected_context = min(successful, key=lambda item: item[0])
    selected_entry = profile[selected_profile_index]
    selected_forward = hazard_forward(selected_result.x, selected_context)

    sample_ages_ms = np.array([0, dt_ms, 0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500])
    sample_age_matrix = sample_ages_ms[None, :]
    recovery_sample = recovery_and_derivatives(
        sample_age_matrix,
        float(selected_entry["abs_ref_ms"]),
        selected_result.x[-2],
        selected_result.x[-1],
    )[0].ravel()

    held_out_delta = None
    if baseline_held_out is not None and selected_entry["held_out"] is not None:
        held_out_delta = (
            baseline_held_out["negative_log_likelihood"]
            - selected_entry["held_out"]["negative_log_likelihood"]
        )

    valid_entries = [entry for entry in profile if entry.get("valid")]
    training_objectives = np.array(
        [entry["penalized_training_objective"] for entry in valid_entries], dtype=np.float64
    )
    held_out_entries = [entry for entry in valid_entries if entry.get("held_out") is not None]
    held_out_best = None
    held_out_span = None
    if held_out_entries:
        held_out_nlls = np.array(
            [entry["held_out"]["negative_log_likelihood"] for entry in held_out_entries],
            dtype=np.float64,
        )
        best_held_out_index = int(np.argmin(held_out_nlls))
        held_out_best = {
            "abs_ref_ms": held_out_entries[best_held_out_index]["abs_ref_ms"],
            "negative_log_likelihood": finite_float(held_out_nlls[best_held_out_index]),
        }
        held_out_span = float(np.max(held_out_nlls) - np.min(held_out_nlls))

    cell_warnings: list[str] = []
    if np.isclose(selected_entry["abs_ref_ms"], max(entry["abs_ref_ms"] for entry in valid_entries)):
        cell_warnings.append(
            "Training selected the largest valid abs_ref profile value; the shortest observed training ISI bounds the estimate."
        )
    objective_span = float(np.max(training_objectives) - np.min(training_objectives))
    if objective_span < 1.0:
        cell_warnings.append(
            "The abs_ref training profile spans less than one nat and is weakly identified."
        )
    if held_out_span is not None and held_out_span < 1.0:
        cell_warnings.append(
            "Held-out likelihood changes by less than one nat across valid abs_ref values."
        )
    if held_out_delta is not None and held_out_delta <= 0:
        cell_warnings.append(
            "The selected refractory model does not improve held-out likelihood over the baseline-only model."
        )

    return {
        "cell_id": int(cell_id),
        "trial_count": int(trials),
        "train_trial_indices_one_based": (train_indices + 1).tolist(),
        "held_out_trial_indices_one_based": (held_out_indices + 1).tolist(),
        "total_spikes": int(np.sum(spikes)),
        "minimum_training_complete_isi_ms": finite_float(minimum_train_isi_ms),
        "baseline_only": {
            "optimizer_success": bool(baseline_result.success),
            "optimizer_message": str(baseline_result.message),
            "iterations": int(baseline_result.nit),
            "training": baseline_train,
            "held_out": baseline_held_out,
        },
        "profile": profile,
        "profile_summary": {
            "requested_value_count": int(len(profile)),
            "valid_value_count": int(len(valid_entries)),
            "penalized_training_objective_span": finite_float(objective_span),
            "held_out_nll_span": finite_float(held_out_span) if held_out_span is not None else None,
            "held_out_best_diagnostic_only": held_out_best,
        },
        "selection_rule": "minimum penalized training objective; held-out trial is not used for selection",
        "selected_profile_index": int(selected_profile_index),
        "selected": {
            **selected_entry,
            "baseline_beta": selected_result.x[:basis_count].tolist(),
            "baseline_hz_summary": numeric_summary(selected_forward["baseline_hz"]),
            "recovery_sample": {
                "age_ms": sample_ages_ms.tolist(),
                "probability_multiplier": recovery_sample.tolist(),
            },
            "held_out_nll_improvement_over_baseline_only": finite_float(held_out_delta)
            if held_out_delta is not None
            else None,
        },
        "warnings": cell_warnings,
    }


def load_selected_targets(config: dict[str, Any], cells: list[int]) -> np.ndarray:
    target_config = copy.deepcopy(config)
    target_config["simulation"]["device"] = "cpu"
    target_config["simulation"]["cell_targets"] = cells
    loaded = data_handler.load_gt_data(target_config)
    return loaded["raster_holder"].detach().cpu().numpy().astype(bool, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "simulation_config.yaml",
        help="Production simulation YAML.",
    )
    parser.add_argument("--cells", default="7", help="Cell IDs, e.g. 7 or 1,7,10-12.")
    parser.add_argument(
        "--abs-ref-grid-ms",
        default="0:10:0.5",
        help="Discrete grid as start:stop:step or comma-separated values.",
    )
    parser.add_argument(
        "--held-out-trial",
        type=int,
        default=10,
        help="One-based held-out trial; use 0 to fit all trials (default: 10).",
    )
    parser.add_argument("--baseline-basis", type=int, default=12)
    parser.add_argument("--smoothness", type=float, default=100.0)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--minimum-recovery-ms", type=float, default=0.05)
    parser.add_argument("--maximum-recovery-ms", type=float, default=500.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    cells = parse_cells(args.cells)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    configured_cells = set(int(cell) for cell in config["simulation"]["cell_targets"])
    missing = [cell for cell in cells if cell not in configured_cells]
    if missing:
        raise ValueError(f"Selected cells are not present in the production config: {missing}")

    dt_ms = float(config["simulation"]["dt"])
    requested_grid = parse_grid(args.abs_ref_grid_ms)
    abs_ref_grid, snapped_grid = snap_grid_to_dt(requested_grid, dt_ms)
    if args.minimum_recovery_ms <= 0 or args.maximum_recovery_ms <= args.minimum_recovery_ms:
        raise ValueError("Recovery bounds must be positive and ordered.")

    self_check = finite_difference_self_check()
    if not self_check["passed"]:
        raise RuntimeError(f"Analytic gradient self-check failed: {self_check}")

    raster = load_selected_targets(config, cells)
    cell_reports = []
    for cell_axis, cell_id in enumerate(cells):
        print(f"Fitting refractory hazard for cell {cell_id}...")
        report = fit_cell(
            cell_id=cell_id,
            spikes=raster[cell_axis],
            dt_ms=dt_ms,
            abs_ref_grid_ms=abs_ref_grid,
            held_out_trial_one_based=args.held_out_trial,
            basis_count=args.baseline_basis,
            smoothness=args.smoothness,
            maximum_iterations=args.maxiter,
            minimum_recovery_ms=args.minimum_recovery_ms,
            maximum_recovery_ms=args.maximum_recovery_ms,
        )
        cell_reports.append(report)
        selected = report["selected"]
        print(
            f"  selected abs_ref={selected['abs_ref_ms']:.3f} ms, "
            f"tau50={selected['tau50_ms']:.3f} ms, width={selected['width_ms']:.3f} ms"
        )

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "name": "time-inhomogeneous discrete conditional hazard",
            "likelihood": "full Bernoulli event/no-event likelihood",
            "right_censoring": "included through all no-event bins to trial end",
            "trial_start": "fully recovered because previous spike is unobserved",
            "baseline": "regularized low-dimensional piecewise-linear log-intensity basis",
            "recovery": "hard absolute refractory plus normalized sigmoid tending to one",
            "absolute_refractory_optimization": "discrete profile on the simulation dt grid",
            "selection": "training objective only; held-out likelihood is evaluation only",
        },
        "inputs": {
            "config_path": str(config_path),
            "production_target_loader": "Data.data_handler.load_gt_data",
            "cells": cells,
            "dt_ms": dt_ms,
            "requested_abs_ref_grid_ms": requested_grid.tolist(),
            "profiled_abs_ref_grid_ms": abs_ref_grid.tolist(),
            "grid_values_snapped_to_dt": snapped_grid,
            "held_out_trial_one_based": int(args.held_out_trial),
            "baseline_basis_count": int(args.baseline_basis),
            "smoothness": float(args.smoothness),
            "maximum_iterations": int(args.maxiter),
            "recovery_bounds_ms": [
                float(args.minimum_recovery_ms),
                float(args.maximum_recovery_ms),
            ],
        },
        "analytic_gradient_self_check": self_check,
        "cells": cell_reports,
        "caveats": [
            "This is a statistical conditional-hazard fit, not the exact marginal likelihood of the stochastic LIF simulator.",
            "A smooth time baseline mitigates stimulus nonstationarity but cannot capture every latent voltage or adaptation effect.",
            "The normalized recovery curve fixes its asymptote at one; rel_ref_c is mapped to 0.5 because a free asymptote is confounded with baseline intensity.",
            "Only one held-out trial is used by default; use trial-level resampling or repeated held-out fits for uncertainty.",
            "The current simulator's fake -30-step initial spike must be fixed before transferring these parameters.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote refractory hazard fit: {output_path}")


if __name__ == "__main__":
    main()
