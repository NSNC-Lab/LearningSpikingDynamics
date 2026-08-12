"""One-parameter loss landscapes for the archived local-eligibility model.

This script deliberately imports the frozen pre-dPSC snapshot rather than the
active project.  For each learnable parameter it varies that parameter alone,
keeps the remaining eleven parameters fixed at selected saved-output tracker
values, and runs the full 2.9801 s forward model. Eligibility-based gradient
estimates are accumulated by the archived code without applying an optimizer
update.

Stochastic comparisons use common random numbers: every value in a seed sees
the same uniforms for onset spikes, offset spikes, spontaneous spikes, and the
probabilistic output-spike gate.  Three independent seeds are then aggregated.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SNAPSHOT_ROOT = REPO_ROOT / "Archive" / "pre_dpsc_local_eligibility_snapshot_2026-07-09_1439"
CHECKPOINT = REPO_ROOT / "100_epoch_post_strf_fix_4.mat"

OUTPUT_DIR = HERE / "output"
DATA_DIR = OUTPUT_DIR / "data"
PLOT_DIR = OUTPUT_DIR / "plots"

CELLS = [1, 2, 7]
# Zero-based saved-output batches selected by the lowest saved PSTH SSE for each
# cell in 100_epoch_post_strf_fix_4.mat.
BEST_BATCHES = [6, 3, 3]
DEFAULT_SEEDS = [123, 456, 789]


# Use the exact parameter keys expected by Architecture_Declaration.py.
PARAMETERS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "Strf_gain",
            {
                "label": "STRF gain",
                "objective": "PSTH SSE",
                "grid": np.geomspace(0.001, 0.15, 13),
                "log_x": True,
            },
        ),
        (
            "Strf_alpha",
            {
                "label": "STRF alpha (ms)",
                "objective": "PSTH SSE",
                "grid": np.geomspace(0.1, 250.0, 13),
                "log_x": True,
            },
        ),
        (
            "output_ad",
            {
                "label": "Output adaptation",
                "objective": "PSTH SSE",
                "grid": np.array([0.0, 0.00025, 0.0005, 0.001, 0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.025]),
                "log_x": False,
            },
        ),
        (
            "on_ron_gSYN",
            {
                "label": "on -> ron gSYN",
                "objective": "PSTH SSE",
                "grid": np.geomspace(0.001, 0.15, 13),
                "log_x": True,
            },
        ),
        (
            "off_ron_gSYN",
            {
                "label": "off -> ron gSYN",
                "objective": "PSTH SSE",
                "grid": np.geomspace(0.001, 0.15, 13),
                "log_x": True,
            },
        ),
        (
            "on_sonoff_gSYN",
            {
                "label": "on -> sonoff gSYN",
                "objective": "PSTH SSE",
                "grid": np.geomspace(0.001, 0.15, 13),
                "log_x": True,
            },
        ),
        (
            "off_sonoff_gSYN",
            {
                "label": "off -> sonoff gSYN",
                "objective": "PSTH SSE",
                "grid": np.geomspace(0.001, 0.15, 13),
                "log_x": True,
            },
        ),
        (
            "sonoff_ron_gSYN",
            {
                "label": "sonoff -> ron gSYN",
                "objective": "PSTH SSE",
                "grid": np.geomspace(0.001, 0.15, 13),
                "log_x": True,
            },
        ),
        (
            "abs_ref",
            {
                "label": "Absolute refractory period (ms)",
                "objective": "CV squared error",
                "grid": np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 25.0]),
                "log_x": False,
            },
        ),
        (
            "rel_ref_a",
            {
                "label": "Relative refractory a",
                "objective": "CV squared error",
                "grid": np.array([0.001, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]),
                "log_x": True,
            },
        ),
        (
            "rel_ref_b",
            {
                "label": "Relative refractory b",
                "objective": "CV squared error",
                "grid": np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0]),
                "log_x": False,
            },
        ),
        (
            "rel_ref_c",
            {
                "label": "Relative refractory c",
                "objective": "CV squared error",
                "grid": np.linspace(0.2, 1.0, 11),
                "log_x": False,
            },
        ),
    ]
)


def import_snapshot_modules() -> tuple[Any, ...]:
    """Import only the frozen snapshot, even when run from the active repo."""
    sys.path.insert(0, str(SNAPSHOT_ROOT))
    from Data import data_handler
    from Pre_Processing import preprocess_handler
    from Simulation import (
        Architecture_Declaration,
        Eligibility_handler,
        Loss_handler,
        conditional_handler,
        initialize_from_mat,
        ode_handler,
    )

    loaded_paths = [
        Path(module.__file__).resolve()
        for module in (
            data_handler,
            preprocess_handler,
            Architecture_Declaration,
            Eligibility_handler,
            Loss_handler,
            conditional_handler,
            initialize_from_mat,
            ode_handler,
        )
    ]
    for module_path in loaded_paths:
        if SNAPSHOT_ROOT not in module_path.parents:
            raise RuntimeError(f"Imported active code instead of the snapshot: {module_path}")
    return (
        data_handler,
        preprocess_handler,
        Architecture_Declaration,
        Eligibility_handler,
        Loss_handler,
        conditional_handler,
        initialize_from_mat,
        ode_handler,
    )


def make_args() -> dict[str, Any]:
    with (SNAPSHOT_ROOT / "simulation_config.yaml").open("r", encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    args = copy.deepcopy(args)
    args["paths"]["stimuli"] = str((REPO_ROOT / "Data" / "Targets").resolve())
    args["paths"]["data"] = str(
        (REPO_ROOT / "Data" / "Data" / "all_units_info_with_polished_criteria_modified_perf.mat").resolve()
    )
    args["paths"]["spontaneous_activity"] = str(
        (REPO_ROOT / "Data" / "Data" / "silent_activity_matrix.npy").resolve()
    )
    args["simulation"].update(
        {
            "batch_size": 1,  # Replaced after sweep rows are constructed.
            "cell_targets": CELLS,
            "sim_len": 29801,
            "epochs": 1,
            "device": "cpu",
            "PSTH_granularity": 100,
            "num_params": 12,
        }
    )
    return args


def load_checkpoint_baseline(initialize_from_mat: Any) -> dict[str, np.ndarray]:
    """Load only parameter structs, avoiding the checkpoint's large output array."""
    checkpoint_vars = loadmat(
        CHECKPOINT,
        variable_names=["params"],
        squeeze_me=False,
        struct_as_record=False,
    )
    loaded: dict[str, np.ndarray] | None = None
    # The saved raster was generated before the epoch's final Adam update, and
    # the last tracker slice in `params` records that exact pre-update state.
    # `checkpoint_params` is one Adam step later and is only the continuation
    # state, so it is intentionally not used for this saved-output analysis.
    if "params" in checkpoint_vars:
        loaded = initialize_from_mat.extract_mat_struct_params(checkpoint_vars["params"], 12, 220)
    if loaded is None:
        raise RuntimeError(
            "Could not read the final tracker parameters that generated the saved raster. "
            "The post-Adam continuation checkpoint is not an interchangeable fallback."
        )

    missing = [name for name in PARAMETERS if name not in loaded]
    if missing:
        raise KeyError(f"Checkpoint is missing parameter(s): {missing}")

    baseline: dict[str, np.ndarray] = {}
    for name in PARAMETERS:
        values = np.asarray(loaded[name], dtype=np.float32)
        baseline[name] = np.array(
            [values[batch, cell - 1] for batch, cell in zip(BEST_BATCHES, CELLS)],
            dtype=np.float32,
        )
    return baseline


def build_sweep(baseline: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for name, spec in PARAMETERS.items():
        for index, value in enumerate(np.asarray(spec["grid"], dtype=float)):
            rows.append(
                {
                    "parameter": name,
                    "row_kind": "grid",
                    "grid_index": index,
                    "values": np.repeat(value, len(CELLS)).astype(np.float32),
                }
            )
        rows.append(
            {
                "parameter": name,
                "row_kind": "checkpoint_baseline",
                "grid_index": -1,
                "values": baseline[name].copy(),
            }
        )

    batch_size = len(rows)
    params = {
        name: np.repeat(values[None, :], batch_size, axis=0).astype(np.float32)
        for name, values in baseline.items()
    }
    for row_index, row in enumerate(rows):
        params[row["parameter"]][row_index, :] = row["values"]
        row["row_index"] = row_index
    return params, rows


@contextlib.contextmanager
def common_random_numbers(batch_size: int) -> Iterator[None]:
    """Broadcast one random draw across the parameter-sweep batch axis."""
    original_rand = torch.rand

    def shared_rand(*size: Any, **kwargs: Any) -> torch.Tensor:
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            shape = tuple(int(v) for v in size[0])
        else:
            shape = tuple(int(v) for v in size)

        if len(shape) == 4 and shape[1] == batch_size:
            base_shape = (shape[0], 1, shape[2], shape[3])
            return original_rand(base_shape, **kwargs).expand(shape)
        if len(shape) == 3 and shape[0] == batch_size:
            base_shape = (1, shape[1], shape[2])
            return original_rand(base_shape, **kwargs).expand(shape)
        return original_rand(*size, **kwargs)

    torch.rand = shared_rand  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.rand = original_rand  # type: ignore[assignment]


def calculate_cv_objective_and_gradient(
    states: dict[str, Any], gt_raster: torch.Tensor, args: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the archived CV objective, but retain every batch/cell value."""
    spikes = states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
    batch_size = spikes.shape[0]
    num_cells = spikes.shape[2]
    loss = torch.zeros((batch_size, num_cells), dtype=torch.float32)
    gradient = torch.zeros_like(loss)
    sim_cv_holder = torch.full_like(loss, torch.nan)
    valid = torch.zeros((batch_size, num_cells), dtype=torch.bool)

    data_cvs: list[torch.Tensor] = []
    for cell_index in range(num_cells):
        data_isis = []
        for trial in range(10):
            spike_indices = torch.where(gt_raster[cell_index, trial, :] == 1)[0]
            data_isis.append(torch.diff(spike_indices))
        concatenated = torch.cat(data_isis).to(torch.float32)
        data_cvs.append(torch.std(concatenated) / torch.mean(concatenated))

    for batch_index in range(batch_size):
        for cell_index in range(num_cells):
            sim_isis = []
            trials_with_spikes = 0
            for trial in range(10):
                spike_indices = torch.where(spikes[batch_index, trial, cell_index, :] == 1)[0]
                if len(spike_indices) > 0:
                    trials_with_spikes += 1
                    sim_isis.append(torch.diff(spike_indices))

            # This is the same sparse-spike rule as the archived Loss_handler:
            # the CV objective and its gradient contribute zero when insufficient.
            if trials_with_spikes <= 1:
                continue
            concatenated = torch.cat(sim_isis).to(torch.float32)
            if len(concatenated) <= 1 or torch.mean(concatenated) == 0:
                continue
            sim_cv = torch.std(concatenated) / torch.mean(concatenated)
            data_cv = data_cvs[cell_index]
            sim_cv_holder[batch_index, cell_index] = sim_cv
            valid[batch_index, cell_index] = True
            loss[batch_index, cell_index] = (sim_cv - data_cv) ** 2
            gradient[batch_index, cell_index] = 2 * (sim_cv - data_cv)
    return loss, gradient, sim_cv_holder, valid


def extract_parameter_gradients(states: dict[str, Any]) -> dict[str, np.ndarray]:
    learnable = states["neurons"]["Learnable"]
    gradients = {
        "Strf_gain": learnable["STRF_gain_grad"],
        "Strf_alpha": learnable["STRF_alpha_grad"],
        "output_ad": learnable["ron"]["output_ad_grad"],
        "abs_ref": learnable["ron"]["abs_ref_grad"],
        "rel_ref_a": learnable["ron"]["rel_ref_a_grad"],
        "rel_ref_b": learnable["ron"]["rel_ref_b_grad"],
        "rel_ref_c": learnable["ron"]["rel_ref_c_grad"],
    }
    for synapse in ("on_ron", "off_ron", "on_sonoff", "off_sonoff", "sonoff_ron"):
        gradients[f"{synapse}_gSYN"] = states["synapses"]["Learnable"][synapse]["gSYN_grad"]
    return {name: tensor.detach().cpu().numpy().astype(float) for name, tensor in gradients.items()}


def run_seed(
    seed: int,
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    gt_data: dict[str, torch.Tensor],
    modules: tuple[Any, ...],
) -> list[dict[str, Any]]:
    (
        _data_handler,
        preprocess_handler,
        Architecture_Declaration,
        Eligibility_handler,
        Loss_handler,
        conditional_handler,
        _initialize_from_mat,
        ode_handler,
    ) = modules

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    batch_size = len(rows)
    device = torch.device("cpu")
    seed_start = time.perf_counter()

    with torch.no_grad(), common_random_numbers(batch_size):
        states = Architecture_Declaration.build_network(args, device, params)
        prep_start = time.perf_counter()
        preprocessed = preprocess_handler.preprocess(args, states, device)
        print(
            f"Seed {seed}: preprocessing {batch_size} sweep rows took "
            f"{time.perf_counter() - prep_start:.1f} s",
            flush=True,
        )

        psth_loss = torch.zeros((batch_size, len(CELLS)), dtype=torch.float32)
        granularity = int(args["simulation"]["PSTH_granularity"])
        sim_len = int(args["simulation"]["sim_len"])
        loop_start = time.perf_counter()

        for timestep in range(sim_len):
            states = ode_handler.run_odes(args, states, preprocessed["spks"], timestep)
            states = Eligibility_handler.update_eligibility(
                args, states, preprocessed["onset_offset_rates"], timestep
            )
            states = conditional_handler.run_conditionals(args, states, timestep)

            if timestep % granularity == 0 and timestep > 0:
                holder = states["neurons"]["Dynamic"]["ron"]["spikes_holder"][
                    :, :, :, timestep - granularity : timestep
                ]
                sim_psth = torch.sum(torch.sum(holder, dim=1), dim=-1)
                target = gt_data["psth_holder"][None, :, int(timestep / granularity) - 1]
                residual = sim_psth - target
                psth_loss += residual**2
                states = Loss_handler.update_grad(states, 2 * residual)

            if timestep and timestep % 2500 == 0:
                elapsed = time.perf_counter() - loop_start
                rate = timestep / max(elapsed, 1e-9)
                remaining = (sim_len - timestep - 1) / max(rate, 1e-9)
                print(
                    f"Seed {seed}: timestep {timestep}/{sim_len - 1}, "
                    f"elapsed {elapsed:.1f} s, ETA {remaining:.1f} s",
                    flush=True,
                )

        cv_loss, cv_gradient, sim_cv, cv_valid = calculate_cv_objective_and_gradient(
            states, gt_data["raster_holder"], args
        )
        states = Loss_handler.update_grad_CV(states, cv_gradient)

    spike_counts = (
        states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
        .sum(dim=(1, 3))
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    duration_s_per_trial = args["simulation"]["sim_len"] * args["simulation"]["dt"] / 1000.0
    firing_rates = spike_counts / (10.0 * duration_s_per_trial)
    gradient_arrays = extract_parameter_gradients(states)
    psth_np = psth_loss.detach().cpu().numpy().astype(float)
    cv_np = cv_loss.detach().cpu().numpy().astype(float)
    sim_cv_np = sim_cv.detach().cpu().numpy().astype(float)
    cv_valid_np = cv_valid.detach().cpu().numpy()

    records: list[dict[str, Any]] = []
    for row in rows:
        row_index = int(row["row_index"])
        parameter = str(row["parameter"])
        objective_is_psth = PARAMETERS[parameter]["objective"] == "PSTH SSE"
        for cell_index, cell in enumerate(CELLS):
            records.append(
                {
                    "seed": seed,
                    "row_index": row_index,
                    "row_kind": row["row_kind"],
                    "grid_index": row["grid_index"],
                    "parameter": parameter,
                    "parameter_label": PARAMETERS[parameter]["label"],
                    "objective": PARAMETERS[parameter]["objective"],
                    "cell": cell,
                    "checkpoint_batch_zero_based": BEST_BATCHES[cell_index],
                    "parameter_value": float(params[parameter][row_index, cell_index]),
                    "loss": float(psth_np[row_index, cell_index] if objective_is_psth else cv_np[row_index, cell_index]),
                    "psth_sse": float(psth_np[row_index, cell_index]),
                    "cv_squared_error": float(cv_np[row_index, cell_index]),
                    "analytic_gradient": float(gradient_arrays[parameter][row_index, cell_index]),
                    "firing_rate_hz": float(firing_rates[row_index, cell_index]),
                    "spike_count": float(spike_counts[row_index, cell_index]),
                    "sim_cv": float(sim_cv_np[row_index, cell_index]),
                    "cv_valid": bool(cv_valid_np[row_index, cell_index]),
                }
            )

    print(
        f"Seed {seed}: complete in {time.perf_counter() - seed_start:.1f} s",
        flush=True,
    )
    del states, preprocessed, psth_loss, cv_loss, cv_gradient, sim_cv, cv_valid
    gc.collect()
    return records


def same_nonzero_sign(update_signal: float, sampled_slope: float) -> bool:
    """Raw green/red rule requested for each plotted point."""
    return bool(
        np.isfinite(update_signal)
        and np.isfinite(sampled_slope)
        and update_signal * sampled_slope > 0
    )


def omit_ill_conditioned_baseline_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop an extra baseline point when it is nearly a duplicate grid x.

    The raw CSV remains untouched.  This only protects the adjacent-point slope
    calculation from a baseline/grid separation that is tiny relative to the
    intentional grid spacing.
    """
    keep = pd.Series(True, index=raw.index)
    for (_parameter, _cell), curve in raw.groupby(["parameter", "cell"], sort=False):
        grid_values = np.sort(
            curve.loc[curve["row_kind"] == "grid", "parameter_value"].unique().astype(float)
        )
        baseline_values = curve.loc[
            curve["row_kind"] == "checkpoint_baseline", "parameter_value"
        ].unique()
        if len(grid_values) < 2 or len(baseline_values) != 1:
            continue
        typical_gap = float(np.median(np.diff(grid_values)))
        nearest_gap = float(np.min(np.abs(grid_values - float(baseline_values[0]))))
        if nearest_gap <= max(1e-12, 0.02 * typical_gap):
            keep.loc[
                curve.index[curve["row_kind"] == "checkpoint_baseline"]
            ] = False
    return raw.loc[keep].copy()


def aggregate_results(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate sampled adjacent-point slopes and aggregate over seeds.

    Every plotted point retains the requested raw green/red sign comparison.
    A stricter qualified status is also calculated for reporting: endpoints,
    flat/turning secants, undefined CV stencils, and mixed seed directions are
    excluded from the qualified headline rather than counted as successes.
    """
    analysis_raw = omit_ill_conditioned_baseline_rows(raw)
    # Average exact duplicates before differentiating so every x is unique.
    per_seed = (
        analysis_raw.groupby(
            ["seed", "parameter", "parameter_label", "objective", "cell", "parameter_value"],
            as_index=False,
            dropna=False,
        )
        .agg(
            loss=("loss", "mean"),
            analytic_gradient=("analytic_gradient", "mean"),
            psth_sse=("psth_sse", "mean"),
            cv_squared_error=("cv_squared_error", "mean"),
            firing_rate_hz=("firing_rate_hz", "mean"),
            spike_count=("spike_count", "mean"),
            sim_cv=("sim_cv", "mean"),
            cv_valid=("cv_valid", "max"),
        )
    )

    slope_parts: list[pd.DataFrame] = []
    for (_seed, _parameter, _cell), curve in per_seed.groupby(
        ["seed", "parameter", "cell"], sort=False
    ):
        curve = curve.sort_values("parameter_value").copy()
        x = curve["parameter_value"].to_numpy(dtype=float)
        y = curve["loss"].to_numpy(dtype=float)
        if len(curve) >= 3:
            slope = np.gradient(y, x, edge_order=1)
        elif len(curve) == 2:
            slope = np.gradient(y, x, edge_order=1)
        else:
            slope = np.array([np.nan])
        left_secant = np.full(len(curve), np.nan, dtype=float)
        right_secant = np.full(len(curve), np.nan, dtype=float)
        if len(curve) >= 2:
            secants = np.diff(y) / np.diff(x)
            left_secant[1:] = secants
            right_secant[:-1] = secants
        interior = np.arange(len(curve)) > 0
        interior &= np.arange(len(curve)) < len(curve) - 1

        update = curve["analytic_gradient"].to_numpy(dtype=float)
        update_tolerance = 1e-10 * max(float(np.nanmax(np.abs(update))), 1.0)
        secant_tolerance = 1e-10 * max(
            float(np.nanmax(np.abs(np.r_[left_secant, right_secant]))), 1.0
        )
        update_nonzero = np.isfinite(update) & (np.abs(update) > update_tolerance)
        secants_nonzero = (
            np.isfinite(left_secant)
            & np.isfinite(right_secant)
            & (np.abs(left_secant) > secant_tolerance)
            & (np.abs(right_secant) > secant_tolerance)
        )
        secants_consistent = secants_nonzero & (left_secant * right_secant > 0)

        objective_is_cv = bool((curve["objective"] == "CV squared error").iloc[0])
        cv_values = curve["cv_valid"].astype(bool).to_numpy()
        cv_stencil_valid = np.ones(len(curve), dtype=bool)
        cv_stencil_invalid = np.zeros(len(curve), dtype=bool)
        if objective_is_cv:
            cv_stencil_valid[:] = False
            for index in range(1, len(curve) - 1):
                cv_stencil_valid[index] = bool(cv_values[index - 1 : index + 2].all())
                cv_stencil_invalid[index] = not cv_stencil_valid[index]

        curve["numerical_slope"] = slope
        curve["left_secant"] = left_secant
        curve["right_secant"] = right_secant
        curve["is_interior"] = interior
        curve["update_nonzero"] = update_nonzero
        curve["secants_consistent"] = secants_consistent
        curve["cv_stencil_valid"] = cv_stencil_valid
        curve["cv_stencil_invalid"] = cv_stencil_invalid
        curve["seed_qualified"] = (
            interior & update_nonzero & secants_consistent & cv_stencil_valid
        )
        curve["seed_aligned"] = [
            same_nonzero_sign(float(g), float(s))
            for g, s in zip(curve["analytic_gradient"], curve["numerical_slope"])
        ]
        slope_parts.append(curve)
    per_seed_slopes = pd.concat(slope_parts, ignore_index=True)

    grouped = per_seed_slopes.groupby(
        ["parameter", "parameter_label", "objective", "cell", "parameter_value"],
        as_index=False,
        dropna=False,
    )
    aggregate = grouped.agg(
        n_seeds=("seed", "nunique"),
        loss_mean=("loss", "mean"),
        loss_std=("loss", "std"),
        gradient_mean=("analytic_gradient", "mean"),
        gradient_std=("analytic_gradient", "std"),
        slope_mean=("numerical_slope", "mean"),
        slope_std=("numerical_slope", "std"),
        seed_alignment_fraction=("seed_aligned", "mean"),
        seed_qualified_fraction=("seed_qualified", "mean"),
        update_positive_fraction=("analytic_gradient", lambda values: float((values > 0).mean())),
        slope_positive_fraction=("numerical_slope", lambda values: float((values > 0).mean())),
        interior_fraction=("is_interior", "mean"),
        update_nonzero_fraction=("update_nonzero", "mean"),
        secant_consistency_fraction=("secants_consistent", "mean"),
        cv_stencil_valid_fraction=("cv_stencil_valid", "mean"),
        cv_stencil_invalid_fraction=("cv_stencil_invalid", "mean"),
        firing_rate_mean_hz=("firing_rate_hz", "mean"),
        firing_rate_std_hz=("firing_rate_hz", "std"),
        spike_count_mean=("spike_count", "mean"),
        sim_cv_mean=("sim_cv", "mean"),
        cv_valid_fraction=("cv_valid", "mean"),
    )
    aggregate["loss_sem"] = aggregate["loss_std"].fillna(0.0) / np.sqrt(aggregate["n_seeds"])
    aggregate["gradient_sem"] = aggregate["gradient_std"].fillna(0.0) / np.sqrt(
        aggregate["n_seeds"]
    )
    aggregate["slope_sem"] = aggregate["slope_std"].fillna(0.0) / np.sqrt(aggregate["n_seeds"])
    aggregate["firing_rate_sem_hz"] = aggregate["firing_rate_std_hz"].fillna(0.0) / np.sqrt(
        aggregate["n_seeds"]
    )

    aggregate["raw_aligned"] = [
        same_nonzero_sign(float(update), float(slope))
        for update, slope in zip(aggregate["gradient_mean"], aggregate["slope_mean"])
    ]
    aggregate["classification"] = np.where(aggregate["raw_aligned"], "green", "red")

    all_seeds_qualified = np.isclose(aggregate["seed_qualified_fraction"], 1.0)
    update_all_positive = np.isclose(aggregate["update_positive_fraction"], 1.0)
    update_all_negative = np.isclose(aggregate["update_positive_fraction"], 0.0)
    slope_all_positive = np.isclose(aggregate["slope_positive_fraction"], 1.0)
    slope_all_negative = np.isclose(aggregate["slope_positive_fraction"], 0.0)
    stable_green = all_seeds_qualified & (
        (update_all_positive & slope_all_positive)
        | (update_all_negative & slope_all_negative)
    )
    stable_red = all_seeds_qualified & (
        (update_all_positive & slope_all_negative)
        | (update_all_negative & slope_all_positive)
    )
    invalid_cv = aggregate["cv_stencil_invalid_fraction"] > 0
    aggregate["qualified_status"] = np.select(
        [stable_green, stable_red, invalid_cv],
        ["stable_green", "stable_red", "invalid_cv"],
        default="inconclusive",
    )
    # Backward-compatible alias used by earlier report drafts. It is explicitly
    # the raw face color, not the qualified headline classification.
    aggregate["aligned"] = aggregate["raw_aligned"]
    return per_seed_slopes, aggregate


def nice_number(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e5:
        return f"{value:.3e}"
    return f"{value:.5g}"


def make_plots(aggregate: pd.DataFrame, baseline: dict[str, np.ndarray]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
        }
    )
    green = "#2ca25f"
    red = "#de2d26"

    for parameter, spec in PARAMETERS.items():
        fig, axes = plt.subplots(2, len(CELLS), figsize=(10.4, 6.6), constrained_layout=True)
        fig.suptitle(
            f"{spec['label']}: archived local eligibility versus sampled loss slope",
            fontsize=12,
            fontweight="bold",
        )

        for cell_index, cell in enumerate(CELLS):
            curve = aggregate[
                (aggregate["parameter"] == parameter) & (aggregate["cell"] == cell)
            ].sort_values("parameter_value")
            x = curve["parameter_value"].to_numpy(dtype=float)
            loss = curve["loss_mean"].to_numpy(dtype=float)
            loss_sem = curve["loss_sem"].to_numpy(dtype=float)
            gradient = curve["gradient_mean"].to_numpy(dtype=float)
            slope = curve["slope_mean"].to_numpy(dtype=float)
            gradient_sem = curve["gradient_sem"].to_numpy(dtype=float)
            slope_sem = curve["slope_sem"].to_numpy(dtype=float)
            point_colors = np.where(curve["raw_aligned"].to_numpy(dtype=bool), green, red)
            status = curve["qualified_status"].to_numpy(dtype=str)
            invalid = status == "invalid_cv"
            inconclusive = status == "inconclusive"

            ax_loss = axes[0, cell_index]
            ax_loss.plot(x, loss, color="#353535", linewidth=1.2, zorder=1)
            ax_loss.fill_between(x, loss - loss_sem, loss + loss_sem, color="#9ecae1", alpha=0.35)
            ax_loss.scatter(
                x, loss, c=point_colors, s=34, edgecolor="white", linewidth=0.6, zorder=3
            )
            if inconclusive.any():
                ax_loss.scatter(
                    x[inconclusive],
                    loss[inconclusive],
                    facecolors="none",
                    edgecolors="#636363",
                    s=58,
                    linewidth=0.9,
                    zorder=4,
                )
            if invalid.any():
                ax_loss.scatter(
                    x[invalid],
                    loss[invalid],
                    marker="x",
                    color="#525252",
                    s=45,
                    linewidth=1.2,
                    zorder=5,
                )
            ax_loss.axvline(
                float(baseline[parameter][cell_index]),
                color="#756bb1",
                linestyle="--",
                linewidth=1.0,
                label="saved-output value",
            )
            ax_loss.set_title(f"Cell {cell} (saved batch {BEST_BATCHES[cell_index]})")
            ax_loss.set_ylabel(spec["objective"])
            ax_loss.grid(alpha=0.2)
            if spec["log_x"]:
                ax_loss.set_xscale("log")
            ax_loss.legend(loc="lower left", fontsize=6, framealpha=0.82)

            ax_grad = axes[1, cell_index]
            scale = max(float(np.nanmax(np.abs(np.r_[gradient, slope]))), 1e-12)
            ax_grad.axhline(0.0, color="#777777", linewidth=0.7)
            ax_grad.plot(
                x,
                gradient / scale,
                color="#3182bd",
                marker="o",
                markersize=3,
                linewidth=1.1,
                label="archived eligibility update signal",
            )
            ax_grad.fill_between(
                x,
                (gradient - gradient_sem) / scale,
                (gradient + gradient_sem) / scale,
                color="#3182bd",
                alpha=0.15,
            )
            ax_grad.plot(
                x,
                slope / scale,
                color="#f16913",
                marker="s",
                markersize=3,
                linewidth=1.1,
                label="numerical loss slope",
            )
            ax_grad.fill_between(
                x,
                (slope - slope_sem) / scale,
                (slope + slope_sem) / scale,
                color="#f16913",
                alpha=0.15,
            )
            ax_grad.axvline(
                float(baseline[parameter][cell_index]),
                color="#756bb1",
                linestyle="--",
                linewidth=1.0,
            )
            ax_grad.set_ylabel("Signal / curve max abs")
            ax_grad.set_xlabel(spec["label"])
            ax_grad.grid(alpha=0.2)
            if spec["log_x"]:
                ax_grad.set_xscale("log")
            if cell_index == 0:
                ax_grad.legend(loc="best")

            stable_matches = int((status == "stable_green").sum())
            qualified_total = int(np.isin(status, ["stable_green", "stable_red"]).sum())
            unresolved = int(np.isin(status, ["inconclusive", "invalid_cv"]).sum())
            ax_loss.text(
                0.02,
                0.96,
                f"qualified green: {stable_matches}/{qualified_total}; unresolved: {unresolved}",
                transform=ax_loss.transAxes,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
            )

        fig.savefig(PLOT_DIR / f"{parameter}_landscape.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def build_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter, spec in PARAMETERS.items():
        subset = aggregate[aggregate["parameter"] == parameter]
        status = subset["qualified_status"]
        stable_green = int((status == "stable_green").sum())
        stable_red = int((status == "stable_red").sum())
        qualified = stable_green + stable_red
        row: dict[str, Any] = {
            "parameter": parameter,
            "parameter_label": spec["label"],
            "objective": spec["objective"],
            "raw_alignment_fraction": float(subset["raw_aligned"].mean()),
            "raw_points": int(len(subset)),
            "raw_aligned_points": int(subset["raw_aligned"].sum()),
            "stable_green_points": stable_green,
            "stable_red_points": stable_red,
            "qualified_points": qualified,
            "qualified_alignment_fraction": (
                float(stable_green / qualified) if qualified else float("nan")
            ),
            "inconclusive_points": int((status == "inconclusive").sum()),
            "invalid_cv_points": int((status == "invalid_cv").sum()),
        }
        for cell in CELLS:
            cell_subset = subset[subset["cell"] == cell]
            cell_status = cell_subset["qualified_status"]
            cell_green = int((cell_status == "stable_green").sum())
            cell_red = int((cell_status == "stable_red").sum())
            cell_qualified = cell_green + cell_red
            row[f"cell_{cell}_qualified_alignment_fraction"] = (
                float(cell_green / cell_qualified) if cell_qualified else float("nan")
            )
            row[f"cell_{cell}_stable_green_points"] = cell_green
            row[f"cell_{cell}_stable_red_points"] = cell_red
            row[f"cell_{cell}_qualified_points"] = cell_qualified
        rows.append(row)
    return pd.DataFrame(rows)


def validate_raw_results(
    raw: pd.DataFrame, requested_seeds: list[int], rows: list[dict[str, Any]]
) -> None:
    """Reject partial/stale reuse and verify the common-random-number control."""
    expected_seeds = sorted(int(seed) for seed in requested_seeds)
    observed_seeds = sorted(int(seed) for seed in raw["seed"].unique())
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Raw-result seeds {observed_seeds} do not match requested seeds {expected_seeds}."
        )
    expected_count = len(expected_seeds) * len(rows) * len(CELLS)
    if len(raw) != expected_count:
        raise ValueError(f"Expected {expected_count} raw rows, found {len(raw)}.")
    key_counts = raw.groupby(["seed", "row_index", "cell"]).size()
    if len(key_counts) != expected_count or not (key_counts == 1).all():
        raise ValueError("Raw results are incomplete or contain duplicate seed/row/cell records.")

    baselines = raw[raw["row_kind"] == "checkpoint_baseline"]
    control_columns = ["psth_sse", "cv_squared_error", "firing_rate_hz", "spike_count"]
    for (seed, cell), control in baselines.groupby(["seed", "cell"]):
        for column in control_columns:
            values = control[column].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=0.0, atol=0.0, equal_nan=True):
                raise RuntimeError(
                    f"Common-random-number control failed for seed={seed}, cell={cell}, "
                    f"column={column}."
                )

    row_lookup = {int(row["row_index"]): row for row in rows}
    for _, record in raw.iterrows():
        row = row_lookup[int(record["row_index"])]
        cell_index = CELLS.index(int(record["cell"]))
        expected_value = float(row["values"][cell_index])
        observed_value = float(record["parameter_value"])
        if str(record["parameter"]) != str(row["parameter"]):
            raise ValueError(
                f"Raw row {record['row_index']} parameter {record['parameter']} does not "
                f"match the current sweep parameter {row['parameter']}."
            )
        parameter_spec = PARAMETERS[str(row["parameter"])]
        if str(record["parameter_label"]) != str(parameter_spec["label"]):
            raise ValueError(
                f"Raw row {record['row_index']} label {record['parameter_label']} does not "
                f"match the current parameter label {parameter_spec['label']}."
            )
        if str(record["objective"]) != str(parameter_spec["objective"]):
            raise ValueError(
                f"Raw row {record['row_index']} objective {record['objective']} does not "
                f"match the current objective {parameter_spec['objective']}."
            )
        if str(record["row_kind"]) != str(row["row_kind"]):
            raise ValueError(
                f"Raw row {record['row_index']} kind {record['row_kind']} does not match "
                f"the current sweep kind {row['row_kind']}."
            )
        if int(record["grid_index"]) != int(row["grid_index"]):
            raise ValueError(
                f"Raw row {record['row_index']} grid index does not match the current sweep."
            )
        if not np.isclose(observed_value, expected_value, rtol=1e-7, atol=1e-12):
            raise ValueError(
                "Raw parameter values do not match the current grid/final tracker values: "
                f"row={record['row_index']}, cell={record['cell']}, "
                f"observed={observed_value}, expected={expected_value}."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Independent stochastic seeds (common random numbers within each seed).",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Skip the simulation and rebuild aggregation/plots from output/data/raw_results.csv.",
    )
    parsed = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    modules = import_snapshot_modules()
    (
        data_handler,
        _preprocess_handler,
        _Architecture_Declaration,
        _Eligibility_handler,
        _Loss_handler,
        _conditional_handler,
        initialize_from_mat,
        _ode_handler,
    ) = modules
    args = make_args()
    baseline = load_checkpoint_baseline(initialize_from_mat)
    params, rows = build_sweep(baseline)
    args["simulation"]["batch_size"] = len(rows)

    raw_path = DATA_DIR / "raw_results.csv"
    run_start = time.perf_counter()
    if parsed.reuse_raw:
        raw = pd.read_csv(raw_path)
    else:
        print(
            f"Running {len(rows)} one-parameter sweep rows x {len(CELLS)} cells x "
            f"{len(parsed.seeds)} seeds with the archived snapshot.",
            flush=True,
        )
        print(f"Snapshot: {SNAPSHOT_ROOT}", flush=True)
        print(f"Checkpoint: {CHECKPOINT}", flush=True)
        gt_data = data_handler.load_gt_data(args)
        all_records: list[dict[str, Any]] = []
        for seed in parsed.seeds:
            all_records.extend(run_seed(seed, args, params, rows, gt_data, modules))
            pd.DataFrame(all_records).to_csv(raw_path, index=False)
        raw = pd.DataFrame(all_records)
        raw.to_csv(raw_path, index=False)

    validate_raw_results(raw, parsed.seeds, rows)
    per_seed, aggregate = aggregate_results(raw)
    summary = build_summary(aggregate)
    per_seed.to_csv(DATA_DIR / "per_seed_slopes.csv", index=False)
    aggregate.to_csv(DATA_DIR / "aggregate_results.csv", index=False)
    summary.to_csv(DATA_DIR / "alignment_summary.csv", index=False)
    make_plots(aggregate, baseline)

    metadata = {
        "analysis_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_root": str(SNAPSHOT_ROOT),
        "checkpoint": str(CHECKPOINT),
        "cells": CELLS,
        "checkpoint_batches_zero_based": BEST_BATCHES,
        "seeds": sorted(int(seed) for seed in raw["seed"].unique()),
        "sim_len": args["simulation"]["sim_len"],
        "dt_ms": args["simulation"]["dt"],
        "psth_granularity_steps": args["simulation"]["PSTH_granularity"],
        "sweep_rows": len(rows),
        "classification": (
            "Every retained plotted point has the requested raw face color: green only when the mean archived "
            "eligibility-based update signal and mean adjacent-point sampled loss slope have "
            "the same nonzero sign, red otherwise. A qualified stable-green or stable-red point "
            "must be interior, have nonzero same-direction left/right secants, have a valid CV "
            "stencil when applicable, and have unanimous update-signal signs and unanimous "
            "slope signs across every seed. Other "
            "points are excluded from the qualified headline."
        ),
        "common_random_numbers": (
            "Within each seed, onset, offset, spontaneous, and output-gate random draws are "
            "shared across all parameter values."
        ),
        "objective_mapping": {name: spec["objective"] for name, spec in PARAMETERS.items()},
        "baseline": {name: [float(v) for v in values] for name, values in baseline.items()},
        "grid": {name: [float(v) for v in spec["grid"]] for name, spec in PARAMETERS.items()},
        "elapsed_seconds": time.perf_counter() - run_start,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "notes": [
            "No Adam or other parameter update was applied.",
            "Each non-baseline sweep changes at most one parameter; all other parameters remain at the selected final-tracker values.",
            "The archived ordering and PSTH-bin convention are preserved, including that eligibility is updated at the bin-closing timestep while the sampled spike count excludes that timestep.",
            "Refractory parameters are compared with the archived CV squared-error objective; all other parameters use summed PSTH SSE.",
            "When the archived CV code considers a simulation too sparse for a CV, both its CV loss contribution and eligibility-based update signal are recorded as zero.",
            "This is an exploratory consistency check of a surrogate update signal against a coarse sampled hard-spike loss curve, not a proof of an exact derivative.",
            "Best batches were selected in-sample by saved PSTH SSE; this report makes no held-out or generalization claim.",
        ],
    }
    with (DATA_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    print("\nQualified directional-consistency summary:", flush=True)
    for _, row in summary.iterrows():
        qualified = int(row["qualified_points"])
        green = int(row["stable_green_points"])
        fraction = 100.0 * float(row["qualified_alignment_fraction"]) if qualified else float("nan")
        print(
            f"  {row['parameter']:<18} {green:>3}/{qualified:<3} "
            f"({fraction:.1f}%), inconclusive={int(row['inconclusive_points'])}, "
            f"invalid_cv={int(row['invalid_cv_points'])}",
            flush=True,
        )
    print(f"Raw results: {raw_path}", flush=True)
    print(f"Aggregate results: {DATA_DIR / 'aggregate_results.csv'}", flush=True)
    print(f"Plots: {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    main()
