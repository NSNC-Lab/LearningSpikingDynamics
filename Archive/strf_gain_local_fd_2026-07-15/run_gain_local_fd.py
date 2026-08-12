"""Local finite-difference validation for archived STRF-gain eligibility.

This is an archive-only analysis of the frozen pre-dPSC July 8/9 model.  It
samples complete 12-parameter vectors from the archived initialization ranges,
then changes only STRF gain within each sampled vector.  Symmetric
multiplicative perturbations share random input/output draws within a sampled
center, while different centers receive independent torch random streams.

The hard-spike loss for one random draw is discontinuous.  Consequently, the
central difference computed here is described as an estimate of the slope of
the expected sampled PSTH SSE, not as an exact derivative of one trajectory.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE_ANALYSIS = REPO_ROOT / "Archive" / "parameter_loss_landscape_2026-07-15" / "run_landscape.py"
DEFAULT_OUTPUT = HERE / "output"

CELLS = [1, 2, 7]
DEFAULT_SEEDS = [123, 456, 789, 1011, 1213, 1415, 1617, 1819]
DEFAULT_NUM_INITS = 8
DEFAULT_INIT_SEED = 20260715
LOG_EPSILONS = np.array([0.0025, 0.005, 0.01, 0.02, 0.05, 0.10], dtype=float)
PRIMARY_LOG_EPSILON = 0.02

# These are the active linear-uniform ranges in the frozen July 8/9
# Simulation/Parameter_initialization.py.  Gain centers are sampled from a
# slightly truncated lower range so the -10% log probe stays above 0.001.
PARAMETER_RANGES: OrderedDict[str, tuple[float, float]] = OrderedDict(
    [
        ("Strf_gain", (0.001, 0.02)),
        ("Strf_alpha", (1.0, 100.0)),
        ("output_ad", (0.0001, 0.005)),
        ("on_ron_gSYN", (0.01, 0.05)),
        ("off_ron_gSYN", (0.01, 0.05)),
        ("on_sonoff_gSYN", (0.01, 0.05)),
        ("off_sonoff_gSYN", (0.01, 0.05)),
        ("sonoff_ron_gSYN", (0.01, 0.05)),
        ("abs_ref", (0.1, 10.0)),
        ("rel_ref_a", (0.001, 10.0)),
        ("rel_ref_b", (0.0, 10.0)),
        ("rel_ref_c", (0.2, 1.0)),
    ]
)

STATUS_COLORS = {
    "green": "#2ca25f",
    "red": "#de2d26",
    "gray": "#9e9e9e",
}


def load_harness() -> Any:
    """Load the already-audited archived simulation harness by file path."""
    spec = importlib.util.spec_from_file_location("archived_landscape_harness", SOURCE_ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analysis harness: {SOURCE_ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_centers(num_inits: int, init_seed: int) -> dict[str, np.ndarray]:
    """Scrambled Latin-hypercube centers, independently stratified per cell."""
    names = list(PARAMETER_RANGES)
    lows = np.array([PARAMETER_RANGES[name][0] for name in names], dtype=float)
    highs = np.array([PARAMETER_RANGES[name][1] for name in names], dtype=float)
    gain_index = names.index("Strf_gain")
    lows[gain_index] = max(lows[gain_index], 0.001 * math.exp(float(LOG_EPSILONS.max())))

    centers = {
        name: np.zeros((num_inits, len(CELLS)), dtype=np.float32)
        for name in names
    }
    for cell_index, cell in enumerate(CELLS):
        sampler = qmc.LatinHypercube(d=len(names), scramble=True, seed=init_seed + 1009 * cell)
        unit = sampler.random(n=num_inits)
        scaled = qmc.scale(unit, lows, highs)
        for param_index, name in enumerate(names):
            centers[name][:, cell_index] = scaled[:, param_index].astype(np.float32)

    minimum_probe = centers["Strf_gain"] * np.exp(-float(LOG_EPSILONS.max()))
    if np.any(minimum_probe < 0.001):
        raise RuntimeError("A sampled gain center produces a perturbation below the gain bound.")
    return centers


def centers_frame(centers: dict[str, np.ndarray]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for init_id in range(centers["Strf_gain"].shape[0]):
        for cell_index, cell in enumerate(CELLS):
            record: dict[str, Any] = {"init_id": init_id, "cell": cell}
            for name in PARAMETER_RANGES:
                record[name] = float(centers[name][init_id, cell_index])
            records.append(record)
    return pd.DataFrame(records)


def build_design(
    centers: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], np.ndarray]:
    """Create center-major rows and a duplicate-center CRN control."""
    offsets = np.concatenate((-LOG_EPSILONS[::-1], [0.0], LOG_EPSILONS))
    rows: list[dict[str, Any]] = []
    for init_id in range(centers["Strf_gain"].shape[0]):
        for offset_index, log_offset in enumerate(offsets):
            if log_offset < 0:
                side = "minus"
            elif log_offset > 0:
                side = "plus"
            else:
                side = "center"
            rows.append(
                {
                    "init_id": init_id,
                    "row_kind": side,
                    "offset_index": offset_index,
                    "log_offset": float(log_offset),
                    "epsilon": abs(float(log_offset)),
                }
            )
        rows.append(
            {
                "init_id": init_id,
                "row_kind": "center_duplicate",
                "offset_index": -1,
                "log_offset": 0.0,
                "epsilon": 0.0,
            }
        )

    batch_size = len(rows)
    params = {
        name: np.zeros((batch_size, len(CELLS)), dtype=np.float32)
        for name in PARAMETER_RANGES
    }
    group_ids = np.zeros(batch_size, dtype=np.int64)
    for row_index, row in enumerate(rows):
        init_id = int(row["init_id"])
        for name in PARAMETER_RANGES:
            params[name][row_index, :] = centers[name][init_id, :]
        params["Strf_gain"][row_index, :] *= np.float32(math.exp(float(row["log_offset"])))
        row["row_index"] = row_index
        group_ids[row_index] = init_id

    for init_id in range(centers["Strf_gain"].shape[0]):
        center_row = next(
            row for row in rows if row["init_id"] == init_id and row["row_kind"] == "center"
        )
        duplicate_row = next(
            row
            for row in rows
            if row["init_id"] == init_id and row["row_kind"] == "center_duplicate"
        )
        for name in PARAMETER_RANGES:
            if not np.array_equal(
                params[name][center_row["row_index"]],
                params[name][duplicate_row["row_index"]],
            ):
                raise RuntimeError(f"Center duplicate differs for {name}, init {init_id}.")
    return params, rows, group_ids


@contextlib.contextmanager
def grouped_common_random_numbers(group_ids: np.ndarray) -> Iterator[None]:
    """Share torch draws within one center, but not between centers."""
    original_rand = torch.rand
    group_ids = np.asarray(group_ids, dtype=np.int64)
    unique_ids, dense_ids = np.unique(group_ids, return_inverse=True)
    batch_size = len(group_ids)
    group_count = len(unique_ids)

    def shared_rand(*size: Any, **kwargs: Any) -> torch.Tensor:
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            shape = tuple(int(value) for value in size[0])
        else:
            shape = tuple(int(value) for value in size)

        if len(shape) == 4 and shape[1] == batch_size:
            base = original_rand((shape[0], group_count, shape[2], shape[3]), **kwargs)
            indices = torch.as_tensor(dense_ids, dtype=torch.long, device=base.device)
            return base.index_select(1, indices)
        if len(shape) == 3 and shape[0] == batch_size:
            base = original_rand((group_count, shape[1], shape[2]), **kwargs)
            indices = torch.as_tensor(dense_ids, dtype=torch.long, device=base.device)
            return base.index_select(0, indices)
        return original_rand(*size, **kwargs)

    torch.rand = shared_rand  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.rand = original_rand  # type: ignore[assignment]


def run_seed(
    seed: int,
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    group_ids: np.ndarray,
    centers: dict[str, np.ndarray],
    gt_data: dict[str, torch.Tensor],
    modules: tuple[Any, ...],
) -> pd.DataFrame:
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

    with torch.no_grad(), grouped_common_random_numbers(group_ids):
        states = Architecture_Declaration.build_network(args, device, params)
        prep_start = time.perf_counter()
        preprocessed = preprocess_handler.preprocess(args, states, device)
        print(
            f"Seed {seed}: preprocessing {batch_size} rows took "
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

            if timestep and timestep % 5000 == 0:
                elapsed = time.perf_counter() - loop_start
                rate = timestep / max(elapsed, 1e-9)
                remaining = (sim_len - timestep - 1) / max(rate, 1e-9)
                print(
                    f"Seed {seed}: timestep {timestep}/{sim_len - 1}, "
                    f"elapsed {elapsed:.1f} s, ETA {remaining:.1f} s",
                    flush=True,
                )

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
    gradients = (
        states["neurons"]["Learnable"]["STRF_gain_grad"]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    losses = psth_loss.detach().cpu().numpy().astype(float)

    records: list[dict[str, Any]] = []
    for row in rows:
        row_index = int(row["row_index"])
        init_id = int(row["init_id"])
        for cell_index, cell in enumerate(CELLS):
            record: dict[str, Any] = {
                    "seed": int(seed),
                    "sim_len": int(args["simulation"]["sim_len"]),
                    "row_index": row_index,
                    "init_id": init_id,
                    "cell": int(cell),
                    "row_kind": str(row["row_kind"]),
                    "offset_index": int(row["offset_index"]),
                    "log_offset": float(row["log_offset"]),
                    "epsilon": float(row["epsilon"]),
                    "center_gain": float(centers["Strf_gain"][init_id, cell_index]),
                    "gain": float(params["Strf_gain"][row_index, cell_index]),
                    "loss": float(losses[row_index, cell_index]),
                    "analytic_gradient_gain": float(gradients[row_index, cell_index]),
                    "analytic_gradient_log_gain": float(
                        params["Strf_gain"][row_index, cell_index]
                        * gradients[row_index, cell_index]
                    ),
                    "firing_rate_hz": float(firing_rates[row_index, cell_index]),
                    "spike_count": float(spike_counts[row_index, cell_index]),
                }
            for name in PARAMETER_RANGES:
                if name != "Strf_gain":
                    record[f"center_{name}"] = float(centers[name][init_id, cell_index])
            records.append(record)

    print(f"Seed {seed}: complete in {time.perf_counter() - seed_start:.1f} s", flush=True)
    del states, preprocessed, psth_loss
    gc.collect()
    return pd.DataFrame(records)


def validate_raw(
    raw: pd.DataFrame,
    seeds: list[int],
    params: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    centers: dict[str, np.ndarray],
    sim_len: int,
) -> None:
    expected = len(seeds) * len(rows) * len(CELLS)
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} raw rows, found {len(raw)}.")
    if sorted(raw["seed"].unique().astype(int).tolist()) != sorted(int(seed) for seed in seeds):
        raise ValueError("Raw seed set does not match the requested seed set.")
    if set(raw["sim_len"].astype(int).unique()) != {int(sim_len)}:
        raise ValueError("Raw simulation length does not match the requested design.")
    keys = raw.groupby(["seed", "row_index", "cell"]).size()
    if len(keys) != expected or not (keys == 1).all():
        raise ValueError("Raw results contain a missing or duplicate seed/row/cell key.")

    row_lookup = {int(row["row_index"]): row for row in rows}
    for record in raw.itertuples(index=False):
        row = row_lookup[int(record.row_index)]
        cell_index = CELLS.index(int(record.cell))
        expected_gain = float(params["Strf_gain"][int(record.row_index), cell_index])
        expected_center = float(centers["Strf_gain"][int(record.init_id), cell_index])
        if int(record.init_id) != int(row["init_id"]):
            raise ValueError("Raw initialization ID does not match the current design.")
        if str(record.row_kind) != str(row["row_kind"]):
            raise ValueError("Raw row kind does not match the current design.")
        if not np.isclose(float(record.log_offset), float(row["log_offset"]), atol=1e-12):
            raise ValueError("Raw log offset does not match the current design.")
        if not np.isclose(float(record.gain), expected_gain, rtol=1e-7, atol=1e-12):
            raise ValueError("Raw gain does not match the current float32 design value.")
        if not np.isclose(float(record.center_gain), expected_center, rtol=1e-7, atol=1e-12):
            raise ValueError("Raw center gain does not match the current sampled center.")
        for name in PARAMETER_RANGES:
            if name == "Strf_gain":
                continue
            column = f"center_{name}"
            if not hasattr(record, column):
                raise ValueError(f"Raw results are missing sampled-center provenance column {column}.")
            observed_center = float(getattr(record, column))
            expected_parameter_center = float(centers[name][int(record.init_id), cell_index])
            if not np.isclose(
                observed_center,
                expected_parameter_center,
                rtol=1e-7,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Raw sampled center for {name} does not match the current LHS design."
                )

    control_columns = [
        "loss",
        "analytic_gradient_gain",
        "analytic_gradient_log_gain",
        "firing_rate_hz",
        "spike_count",
    ]
    for (_seed, _init_id, _cell), group in raw.groupby(["seed", "init_id", "cell"]):
        center = group[group["row_kind"] == "center"]
        duplicate = group[group["row_kind"] == "center_duplicate"]
        if len(center) != 1 or len(duplicate) != 1:
            raise ValueError("Each context must have one center and one center duplicate.")
        for column in control_columns:
            first = float(center[column].iloc[0])
            second = float(duplicate[column].iloc[0])
            if not np.isclose(first, second, rtol=0.0, atol=0.0, equal_nan=True):
                raise RuntimeError(
                    f"Grouped common-random-number control failed for {column}: "
                    f"seed={_seed}, init={_init_id}, cell={_cell}."
                )


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, math.nan, math.nan, math.nan
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    half_width = critical * sem
    return mean, mean - half_width, mean + half_width, sem


def ci_excludes_zero(low: float, high: float) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0))


def build_paired_tables(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = raw[raw["row_kind"] != "center_duplicate"].copy()
    per_seed_records: list[dict[str, Any]] = []
    local_records: list[dict[str, Any]] = []

    for (seed, init_id, cell), context in primary.groupby(["seed", "init_id", "cell"], sort=True):
        center = context[context["row_kind"] == "center"].iloc[0]
        center_loss = float(center["loss"])
        center_analytic_log = float(center["analytic_gradient_log_gain"])
        for row in context.itertuples(index=False):
            local_records.append(
                {
                    "seed": int(seed),
                    "init_id": int(init_id),
                    "cell": int(cell),
                    "log_offset": float(row.log_offset),
                    "epsilon": abs(float(row.log_offset)),
                    "center_gain": float(center.center_gain),
                    "gain": float(row.gain),
                    "loss": float(row.loss),
                    "center_loss": center_loss,
                    "delta_loss": float(row.loss) - center_loss,
                    "center_analytic_log": center_analytic_log,
                    "firing_rate_hz": float(row.firing_rate_hz),
                }
            )

        for epsilon in LOG_EPSILONS:
            minus = context[np.isclose(context["log_offset"], -epsilon, rtol=0.0, atol=1e-12)]
            plus = context[np.isclose(context["log_offset"], epsilon, rtol=0.0, atol=1e-12)]
            if len(minus) != 1 or len(plus) != 1:
                raise ValueError(f"Missing +/- epsilon pair for init={init_id}, cell={cell}.")
            minus_row = minus.iloc[0]
            plus_row = plus.iloc[0]
            gain_minus = float(minus_row["gain"])
            gain_plus = float(plus_row["gain"])
            if not gain_minus < float(center["center_gain"]) < gain_plus:
                raise RuntimeError("Float32 gain probes are not strictly ordered.")
            actual_log_span = math.log(gain_plus / gain_minus)
            fd_log = (float(plus_row["loss"]) - float(minus_row["loss"])) / actual_log_span
            fd_gain = (float(plus_row["loss"]) - float(minus_row["loss"])) / (
                gain_plus - gain_minus
            )
            per_seed_records.append(
                {
                    "seed": int(seed),
                    "init_id": int(init_id),
                    "cell": int(cell),
                    "epsilon": float(epsilon),
                    "center_gain": float(center["center_gain"]),
                    "gain_minus": gain_minus,
                    "gain_plus": gain_plus,
                    "actual_log_span": actual_log_span,
                    "loss_minus": float(minus_row["loss"]),
                    "loss_center": center_loss,
                    "loss_plus": float(plus_row["loss"]),
                    "fd_gradient_gain": fd_gain,
                    "fd_gradient_log_gain": fd_log,
                    "analytic_gradient_gain": float(center["analytic_gradient_gain"]),
                    "analytic_gradient_log_gain": center_analytic_log,
                    "loss_difference_exact_zero": bool(
                        float(plus_row["loss"]) == float(minus_row["loss"])
                    ),
                    "center_firing_rate_hz": float(center["firing_rate_hz"]),
                }
            )

    per_seed = pd.DataFrame(per_seed_records)
    local_per_seed = pd.DataFrame(local_records)

    aggregate_records: list[dict[str, Any]] = []
    for (init_id, cell, epsilon), group in per_seed.groupby(
        ["init_id", "cell", "epsilon"], sort=True
    ):
        fd_mean, fd_low, fd_high, fd_sem = mean_ci(group["fd_gradient_log_gain"].to_numpy())
        an_mean, an_low, an_high, an_sem = mean_ci(
            group["analytic_gradient_log_gain"].to_numpy()
        )
        raw_aligned = bool(
            np.isfinite(fd_mean)
            and np.isfinite(an_mean)
            and fd_mean != 0
            and an_mean != 0
            and fd_mean * an_mean > 0
        )
        aggregate_records.append(
            {
                "init_id": int(init_id),
                "cell": int(cell),
                "epsilon": float(epsilon),
                "epsilon_percent_log": 100.0 * float(epsilon),
                "center_gain": float(group["center_gain"].iloc[0]),
                "seed_count": int(group["seed"].nunique()),
                "fd_log_mean": fd_mean,
                "fd_log_ci_low": fd_low,
                "fd_log_ci_high": fd_high,
                "fd_log_sem": fd_sem,
                "analytic_log_mean": an_mean,
                "analytic_log_ci_low": an_low,
                "analytic_log_ci_high": an_high,
                "analytic_log_sem": an_sem,
                "fd_resolved": ci_excludes_zero(fd_low, fd_high),
                "analytic_resolved": ci_excludes_zero(an_low, an_high),
                "raw_aligned": raw_aligned,
                "zero_difference_fraction": float(group["loss_difference_exact_zero"].mean()),
                "center_firing_rate_mean_hz": float(group["center_firing_rate_hz"].mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_records)

    aggregate["stable_sign_window"] = False
    aggregate["magnitude_stable_window"] = False
    for (_init_id, _cell), index_values in aggregate.groupby(["init_id", "cell"]).groups.items():
        indices = list(index_values)
        curve = aggregate.loc[indices].sort_values("epsilon")
        ordered_indices = curve.index.to_list()
        signs = np.sign(curve["fd_log_mean"].to_numpy(dtype=float))
        resolved = curve["fd_resolved"].to_numpy(dtype=bool)
        stable = np.zeros(len(curve), dtype=bool)
        magnitude_stable = np.zeros(len(curve), dtype=bool)
        ci_lows = curve["fd_log_ci_low"].to_numpy(dtype=float)
        ci_highs = curve["fd_log_ci_high"].to_numpy(dtype=float)
        for start in range(max(0, len(curve) - 2)):
            stop = start + 3
            if stop > len(curve):
                continue
            window_signs = signs[start:stop]
            window_resolved = resolved[start:stop]
            if window_resolved.all() and np.all(window_signs == window_signs[0]) and window_signs[0] != 0:
                stable[start:stop] = True
                if np.max(ci_lows[start:stop]) <= np.min(ci_highs[start:stop]):
                    magnitude_stable[start:stop] = True
        aggregate.loc[ordered_indices, "stable_sign_window"] = stable
        aggregate.loc[ordered_indices, "magnitude_stable_window"] = magnitude_stable

    qualified = aggregate["analytic_resolved"] & aggregate["stable_sign_window"]
    aggregate["qualified"] = qualified
    aggregate["status"] = "gray"
    aggregate.loc[qualified & aggregate["raw_aligned"], "status"] = "green"
    aggregate.loc[qualified & ~aggregate["raw_aligned"], "status"] = "red"

    local_aggregate_records: list[dict[str, Any]] = []
    for (init_id, cell, log_offset), group in local_per_seed.groupby(
        ["init_id", "cell", "log_offset"], sort=True
    ):
        delta_mean, delta_low, delta_high, delta_sem = mean_ci(group["delta_loss"].to_numpy())
        loss_mean, loss_low, loss_high, loss_sem = mean_ci(group["loss"].to_numpy())
        analytic_mean, analytic_low, analytic_high, _ = mean_ci(
            group["center_analytic_log"].to_numpy()
        )
        local_aggregate_records.append(
            {
                "init_id": int(init_id),
                "cell": int(cell),
                "log_offset": float(log_offset),
                "epsilon": abs(float(log_offset)),
                "center_gain": float(group["center_gain"].iloc[0]),
                "gain_mean": float(group["gain"].mean()),
                "loss_mean": loss_mean,
                "loss_ci_low": loss_low,
                "loss_ci_high": loss_high,
                "loss_sem": loss_sem,
                "delta_loss_mean": delta_mean,
                "delta_loss_ci_low": delta_low,
                "delta_loss_ci_high": delta_high,
                "delta_loss_sem": delta_sem,
                "analytic_log_mean": analytic_mean,
                "analytic_log_ci_low": analytic_low,
                "analytic_log_ci_high": analytic_high,
                "firing_rate_mean_hz": float(group["firing_rate_hz"].mean()),
            }
        )
    local_aggregate = pd.DataFrame(local_aggregate_records)
    return per_seed, aggregate, local_aggregate


def safe_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    if kind == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


def build_epsilon_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for epsilon, group in aggregate.groupby("epsilon", sort=True):
        qualified = group[group["qualified"]]
        both_resolved = group[group["fd_resolved"] & group["analytic_resolved"]]
        raw_nonzero = group[
            np.isfinite(group["fd_log_mean"])
            & np.isfinite(group["analytic_log_mean"])
            & (group["fd_log_mean"] != 0)
            & (group["analytic_log_mean"] != 0)
        ]
        analytic = group["analytic_log_mean"].to_numpy(dtype=float)
        finite_difference = group["fd_log_mean"].to_numpy(dtype=float)
        denominator = float(np.dot(analytic, analytic))
        calibration = (
            float(np.dot(analytic, finite_difference) / denominator)
            if denominator > 0
            else math.nan
        )
        norm_product = float(np.linalg.norm(analytic) * np.linalg.norm(finite_difference))
        cosine = (
            float(np.dot(analytic, finite_difference) / norm_product)
            if norm_product > 0
            else math.nan
        )
        records.append(
            {
                "epsilon": float(epsilon),
                "epsilon_percent_log": 100.0 * float(epsilon),
                "contexts": int(len(group)),
                "fd_resolved": int(group["fd_resolved"].sum()),
                "both_resolved": int(len(both_resolved)),
                "stable_sign_window": int(group["stable_sign_window"].sum()),
                "magnitude_stable_window": int(group["magnitude_stable_window"].sum()),
                "qualified": int(len(qualified)),
                "green": int((group["status"] == "green").sum()),
                "red": int((group["status"] == "red").sum()),
                "gray": int((group["status"] == "gray").sum()),
                "qualified_alignment_fraction": (
                    float((qualified["status"] == "green").mean())
                    if len(qualified)
                    else math.nan
                ),
                "both_resolved_alignment_fraction": (
                    float(both_resolved["raw_aligned"].mean())
                    if len(both_resolved)
                    else math.nan
                ),
                "raw_alignment_fraction": (
                    float(raw_nonzero["raw_aligned"].mean()) if len(raw_nonzero) else math.nan
                ),
                "zero_difference_fraction": float(group["zero_difference_fraction"].mean()),
                "pearson_all": safe_correlation(analytic, finite_difference, "pearson"),
                "spearman_all": safe_correlation(analytic, finite_difference, "spearman"),
                "cosine_all": cosine,
                "through_origin_fd_on_analytic": calibration,
                "median_fd_to_analytic_ratio_both_resolved": (
                    float(
                        np.median(
                            both_resolved["fd_log_mean"].to_numpy(dtype=float)
                            / both_resolved["analytic_log_mean"].to_numpy(dtype=float)
                        )
                    )
                    if len(both_resolved)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(records)


def choose_primary_epsilon(summary: pd.DataFrame) -> float:
    """Return the predeclared local scale; all other epsilons are sensitivity checks."""
    if not np.isclose(summary["epsilon"], PRIMARY_LOG_EPSILON).any():
        raise ValueError("The predeclared primary epsilon is absent from the current design.")
    return float(PRIMARY_LOG_EPSILON)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_local_landscapes(
    local_aggregate: pd.DataFrame,
    aggregate: pd.DataFrame,
    plot_dir: Path,
) -> None:
    for cell in CELLS:
        contexts = sorted(local_aggregate[local_aggregate["cell"] == cell]["init_id"].unique())
        columns = 4
        rows_count = int(math.ceil(len(contexts) / columns))
        fig, axes = plt.subplots(rows_count, columns, figsize=(13.2, 3.2 * rows_count), squeeze=False)
        for axis, init_id in zip(axes.ravel(), contexts):
            curve = local_aggregate[
                (local_aggregate["cell"] == cell) & (local_aggregate["init_id"] == init_id)
            ].sort_values("log_offset")
            x = curve["log_offset"].to_numpy(dtype=float) * 100.0
            y = curve["delta_loss_mean"].to_numpy(dtype=float)
            low = curve["delta_loss_ci_low"].to_numpy(dtype=float)
            high = curve["delta_loss_ci_high"].to_numpy(dtype=float)
            analytic = float(curve["analytic_log_mean"].iloc[0])
            axis.plot(x, y, color="#355f7c", linewidth=1.6, zorder=2, label="mean paired loss change")
            axis.fill_between(x, low, high, color="#6baed6", alpha=0.22, linewidth=0)
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.axvline(0, color="#777777", linewidth=0.7)

            status_lookup = aggregate[
                (aggregate["cell"] == cell) & (aggregate["init_id"] == init_id)
            ].set_index("epsilon")["status"].to_dict()
            for point in curve.itertuples(index=False):
                if float(point.log_offset) == 0:
                    axis.scatter([0], [0], s=32, color="#222222", zorder=4)
                else:
                    status = status_lookup.get(abs(float(point.log_offset)), "gray")
                    axis.scatter(
                        [100.0 * float(point.log_offset)],
                        [float(point.delta_loss_mean)],
                        s=28,
                        color=STATUS_COLORS[status],
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=4,
                    )
            gain = float(curve["center_gain"].iloc[0])
            fr = float(curve.loc[np.isclose(curve["log_offset"], 0), "firing_rate_mean_hz"].iloc[0])
            axis.set_title(f"Init {int(init_id)} | g0={gain:.5f} | FR={fr:.2f} Hz")
            eligibility_direction = "increasing loss" if analytic > 0 else "decreasing loss" if analytic < 0 else "zero"
            axis.text(
                0.02,
                0.97,
                f"eligibility predicts: {eligibility_direction}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="#b35806",
            )
            axis.set_xlabel("log gain offset (%)")
            axis.set_ylabel("paired PSTH SSE change")
            axis.grid(alpha=0.18)
        for axis in axes.ravel()[len(contexts) :]:
            axis.axis("off")
        fig.suptitle(
            f"Cell {cell}: local STRF-gain slices at Latin-hypercube parameter contexts",
            fontsize=14,
            y=0.995,
        )
        fig.text(
            0.5,
            0.005,
            "Point color is the qualification of the symmetric +/- pair at that epsilon, not an endpoint-specific test.",
            ha="center",
            fontsize=7,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.025, 1, 0.95))
        fig.savefig(plot_dir / f"local_landscapes_cell_{cell}.png", bbox_inches="tight")
        plt.close(fig)


def plot_fd_convergence(aggregate: pd.DataFrame, plot_dir: Path) -> None:
    for cell in CELLS:
        contexts = sorted(aggregate[aggregate["cell"] == cell]["init_id"].unique())
        columns = 4
        rows_count = int(math.ceil(len(contexts) / columns))
        fig, axes = plt.subplots(rows_count, columns, figsize=(13.2, 3.2 * rows_count), squeeze=False)
        for axis, init_id in zip(axes.ravel(), contexts):
            curve = aggregate[
                (aggregate["cell"] == cell) & (aggregate["init_id"] == init_id)
            ].sort_values("epsilon")
            x = curve["epsilon_percent_log"].to_numpy(dtype=float)
            y = curve["fd_log_mean"].to_numpy(dtype=float)
            yerr = np.vstack(
                (
                    y - curve["fd_log_ci_low"].to_numpy(dtype=float),
                    curve["fd_log_ci_high"].to_numpy(dtype=float) - y,
                )
            )
            analytic = float(curve["analytic_log_mean"].iloc[0])
            analytic_low = float(curve["analytic_log_ci_low"].iloc[0])
            analytic_high = float(curve["analytic_log_ci_high"].iloc[0])
            axis.errorbar(x, y, yerr=yerr, color="#355f7c", marker="o", markersize=3.5, capsize=2)
            axis.axhline(analytic, color="#f16913", linestyle="--", linewidth=1.4)
            axis.axhspan(analytic_low, analytic_high, color="#fdae6b", alpha=0.18)
            for point in curve.itertuples(index=False):
                axis.scatter(
                    [float(point.epsilon_percent_log)],
                    [float(point.fd_log_mean)],
                    s=30,
                    color=STATUS_COLORS[str(point.status)],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                )
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=max(1.0, 0.02 * np.nanmax(np.abs(y))))
            axis.grid(alpha=0.18)
            axis.set_title(f"Init {int(init_id)} | g0={float(curve['center_gain'].iloc[0]):.5f}")
            axis.set_xlabel("log-gain epsilon (%)")
            axis.set_ylabel("d loss / d log(gain)")
        for axis in axes.ravel()[len(contexts) :]:
            axis.axis("off")
        fig.suptitle(
            f"Cell {cell}: central-difference epsilon sensitivity (orange = archived eligibility)",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(plot_dir / f"fd_convergence_cell_{cell}.png", bbox_inches="tight")
        plt.close(fig)


def plot_gradient_scatter(aggregate: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2), squeeze=False)
    markers = {1: "o", 2: "s", 7: "^"}
    for axis, epsilon in zip(axes.ravel(), sorted(aggregate["epsilon"].unique())):
        subset = aggregate[np.isclose(aggregate["epsilon"], epsilon)].copy()
        values = np.concatenate(
            (subset["analytic_log_mean"].to_numpy(dtype=float), subset["fd_log_mean"].to_numpy(dtype=float))
        )
        finite = np.abs(values[np.isfinite(values)])
        limit = float(np.nanmax(finite)) if len(finite) else 1.0
        limit = max(limit, 1.0)
        for cell in CELLS:
            cell_subset = subset[subset["cell"] == cell]
            for status in ("gray", "green", "red"):
                points = cell_subset[cell_subset["status"] == status]
                if points.empty:
                    continue
                axis.scatter(
                    points["analytic_log_mean"],
                    points["fd_log_mean"],
                    marker=markers[cell],
                    s=38,
                    color=STATUS_COLORS[status],
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"cell {cell} {status}",
                )
        axis.plot([-limit, limit], [-limit, limit], "--", color="#555555", linewidth=0.9)
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.axvline(0, color="#888888", linewidth=0.6)
        axis.set_xscale("symlog", linthresh=max(1.0, 0.01 * limit))
        axis.set_yscale("symlog", linthresh=max(1.0, 0.01 * limit))
        axis.set_xlim(-1.15 * limit, 1.15 * limit)
        axis.set_ylim(-1.15 * limit, 1.15 * limit)
        axis.grid(alpha=0.16)
        axis.set_title(f"epsilon = {100 * float(epsilon):g}% in log gain")
        axis.set_xlabel("archived eligibility dL/dlog(g)")
        axis.set_ylabel("paired central FD dL/dlog(g)")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle("STRF-gain eligibility versus local expected-loss slope", fontsize=14, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(plot_dir / "gradient_scatter_by_epsilon.png", bbox_inches="tight")
    plt.close(fig)


def plot_status_heatmap(aggregate: pd.DataFrame, plot_dir: Path) -> None:
    contexts = [(init_id, cell) for cell in CELLS for init_id in sorted(aggregate["init_id"].unique())]
    epsilons = sorted(aggregate["epsilon"].unique())
    matrix = np.zeros((len(contexts), len(epsilons)), dtype=int)
    value_map = {"red": -1, "gray": 0, "green": 1}
    for row_index, (init_id, cell) in enumerate(contexts):
        for column_index, epsilon in enumerate(epsilons):
            match = aggregate[
                (aggregate["init_id"] == init_id)
                & (aggregate["cell"] == cell)
                & np.isclose(aggregate["epsilon"], epsilon)
            ]
            matrix[row_index, column_index] = value_map[str(match["status"].iloc[0])]

    fig, axis = plt.subplots(figsize=(8.2, 8.8))
    cmap = ListedColormap([STATUS_COLORS["red"], STATUS_COLORS["gray"], STATUS_COLORS["green"]])
    axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    axis.set_xticks(range(len(epsilons)), [f"{100 * value:g}%" for value in epsilons])
    axis.set_yticks(range(len(contexts)), [f"cell {cell}, init {init_id}" for init_id, cell in contexts])
    axis.set_xlabel("log-gain epsilon")
    axis.set_title("Qualified directional agreement across local contexts")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            label = {1: "G", 0: "-", -1: "R"}[int(matrix[row_index, column_index])]
            axis.text(column_index, row_index, label, ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "status_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_epsilon_summary(summary: pd.DataFrame, plot_dir: Path) -> None:
    x = summary["epsilon_percent_log"].to_numpy(dtype=float)
    contexts = summary["contexts"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8))

    axes[0, 0].plot(x, summary["fd_resolved"] / contexts, "o-", label="FD CI excludes zero")
    axes[0, 0].plot(
        x,
        summary["stable_sign_window"] / contexts,
        "s-",
        label="3-epsilon resolved stable-sign window",
    )
    axes[0, 0].plot(
        x,
        summary["magnitude_stable_window"] / contexts,
        "d-",
        label="3-epsilon common-CI magnitude window",
    )
    axes[0, 0].plot(
        x,
        summary["qualified"] / contexts,
        "^-",
        label="stable sign + eligibility resolved",
    )
    axes[0, 0].set_ylabel("fraction of contexts")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x, summary["raw_alignment_fraction"], "o-", label="raw mean-sign agreement")
    axes[0, 1].plot(
        x,
        summary["qualified_alignment_fraction"],
        "s-",
        label="qualified agreement",
    )
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_ylabel("directional agreement")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(x, summary["zero_difference_fraction"], "o-", color="#756bb1")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_ylabel("exact-zero paired loss fraction")

    axes[1, 1].plot(x, summary["spearman_all"], "o-", label="Spearman")
    axes[1, 1].plot(x, summary["cosine_all"], "s-", label="cosine alignment")
    axes[1, 1].axhline(0, color="#777777", linewidth=0.7)
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set_ylabel("magnitude/direction association")
    axes[1, 1].legend(frameon=False)

    for axis in axes.ravel():
        axis.set_xscale("log")
        axis.set_xlabel("log-gain epsilon (%)")
        axis.grid(alpha=0.2)
    fig.suptitle("Epsilon sensitivity of the STRF-gain finite-difference test", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_dir / "epsilon_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_calibration(
    aggregate: pd.DataFrame,
    summary: pd.DataFrame,
    plot_dir: Path,
) -> None:
    epsilons = sorted(aggregate["epsilon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    rng = np.random.default_rng(20260715)
    markers = {1: "o", 2: "s", 7: "^"}
    for x_index, epsilon in enumerate(epsilons):
        subset = aggregate[
            np.isclose(aggregate["epsilon"], epsilon)
            & aggregate["fd_resolved"]
            & aggregate["analytic_resolved"]
            & (aggregate["analytic_log_mean"] != 0)
        ].copy()
        subset["ratio"] = subset["fd_log_mean"] / subset["analytic_log_mean"]
        ratios = subset["ratio"].to_numpy(dtype=float)
        if len(ratios):
            axes[0].boxplot(
                [ratios],
                positions=[x_index],
                widths=0.48,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "#d9edf7", "edgecolor": "#3182bd"},
                medianprops={"color": "#08519c", "linewidth": 1.5},
                whiskerprops={"color": "#3182bd"},
                capprops={"color": "#3182bd"},
            )
        for cell in CELLS:
            cell_points = subset[subset["cell"] == cell]
            jitter = rng.normal(0.0, 0.035, len(cell_points))
            axes[0].scatter(
                np.full(len(cell_points), x_index) + jitter,
                cell_points["ratio"],
                marker=markers[cell],
                s=25,
                alpha=0.75,
                label=f"cell {cell}" if x_index == 0 else None,
            )
    axes[0].axhline(1.0, color="#de2d26", linestyle="--", linewidth=1.0, label="equal magnitude")
    axes[0].axhline(0.0, color="#777777", linewidth=0.7)
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_xticks(range(len(epsilons)), [f"{100 * value:g}%" for value in epsilons])
    axes[0].set_xlabel("log-gain epsilon")
    axes[0].set_ylabel("FD / archived eligibility")
    axes[0].set_title("Context-level magnitude ratios (resolved comparisons)")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=7)

    x = summary["epsilon_percent_log"].to_numpy(dtype=float)
    axes[1].plot(
        x,
        summary["through_origin_fd_on_analytic"],
        "o-",
        label="through-origin FD-on-eligibility scale",
    )
    axes[1].plot(
        x,
        summary["median_fd_to_analytic_ratio_both_resolved"],
        "s-",
        label="median resolved FD/eligibility",
    )
    axes[1].axhline(1.0, color="#de2d26", linestyle="--", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("log-gain epsilon (%)")
    axes[1].set_ylabel("multiplicative scale")
    axes[1].set_title("Aggregate magnitude calibration")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(0.98, 0.72))
    fig.suptitle("Magnitude agreement is separate from directional agreement", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plot_dir / "magnitude_calibration.png", bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_landscape(local_aggregate: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(CELLS), figsize=(12.6, 4.2), sharex=True)
    for axis, cell in zip(axes, CELLS):
        normalized_curves: list[np.ndarray] = []
        x_reference: np.ndarray | None = None
        for init_id, curve in local_aggregate[local_aggregate["cell"] == cell].groupby("init_id"):
            curve = curve.sort_values("log_offset")
            x = curve["log_offset"].to_numpy(dtype=float) * 100.0
            y = curve["delta_loss_mean"].to_numpy(dtype=float)
            center_loss = float(
                curve.loc[np.isclose(curve["log_offset"], 0), "loss_mean"].iloc[0]
            )
            scale = max(abs(center_loss), 1.0)
            normalized = 100.0 * y / scale
            axis.plot(x, normalized, color="#9ecae1", alpha=0.55, linewidth=0.9)
            normalized_curves.append(normalized)
            x_reference = x
        stack = np.vstack(normalized_curves)
        median = np.median(stack, axis=0)
        lower = np.quantile(stack, 0.25, axis=0)
        upper = np.quantile(stack, 0.75, axis=0)
        axis.fill_between(x_reference, lower, upper, color="#3182bd", alpha=0.18)
        axis.plot(x_reference, median, color="#08519c", linewidth=2.0, label="median")
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.axvline(0, color="#777777", linewidth=0.7)
        axis.set_title(f"Cell {cell}")
        axis.set_xlabel("log gain offset (%)")
        axis.set_ylabel("loss change (% of center PSTH SSE)")
        axis.grid(alpha=0.18)
    fig.suptitle(
        "Ensemble of conditional local landscapes (loss change relative to each context's center loss)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plot_dir / "ensemble_local_landscapes.png", bbox_inches="tight")
    plt.close(fig)


def make_plots(
    local_aggregate: pd.DataFrame,
    aggregate: pd.DataFrame,
    summary: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    plot_local_landscapes(local_aggregate, aggregate, plot_dir)
    plot_fd_convergence(aggregate, plot_dir)
    plot_gradient_scatter(aggregate, plot_dir)
    plot_status_heatmap(aggregate, plot_dir)
    plot_epsilon_summary(summary, plot_dir)
    plot_magnitude_calibration(aggregate, summary, plot_dir)
    plot_ensemble_landscape(local_aggregate, plot_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num-inits", type=int, default=DEFAULT_NUM_INITS)
    parser.add_argument("--init-seed", type=int, default=DEFAULT_INIT_SEED)
    parser.add_argument("--sim-len", type=int, default=29801)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun seeds even if validated seed CSVs exist.")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Run or validate one seed file and exit without aggregating; useful for parallel workers.",
    )
    parsed = parser.parse_args()

    if parsed.seed_only and len(parsed.seeds) != 1:
        raise ValueError("--seed-only requires exactly one value after --seeds.")
    if int(parsed.num_inits) <= 0:
        raise ValueError("--num-inits must be positive.")
    if not parsed.seeds:
        raise ValueError("At least one stochastic seed is required.")
    if int(parsed.sim_len) <= 0:
        raise ValueError("--sim-len must be positive.")

    output_root = parsed.output_root.resolve()
    data_dir = output_root / "data"
    plot_dir = output_root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    harness = load_harness()
    modules = harness.import_snapshot_modules()
    args = harness.make_args()
    args["simulation"]["sim_len"] = int(parsed.sim_len)
    centers = sample_centers(int(parsed.num_inits), int(parsed.init_seed))
    params, rows, group_ids = build_design(centers)
    args["simulation"]["batch_size"] = len(rows)

    if not parsed.seed_only:
        centers_frame(centers).to_csv(data_dir / "parameter_centers.csv", index=False)
    raw_path = data_dir / "raw_results.csv"
    start = time.perf_counter()

    if parsed.seed_only:
        seed = int(parsed.seeds[0])
        seed_path = data_dir / f"raw_seed_{seed}.csv"
        if seed_path.exists() and not parsed.force:
            candidate = pd.read_csv(seed_path)
            validate_raw(candidate, [seed], params, rows, centers, parsed.sim_len)
            print(f"Seed {seed}: reusing validated seed file {seed_path}", flush=True)
            return
        data_handler = modules[0]
        gt_data = data_handler.load_gt_data(args)
        frame = run_seed(seed, args, params, rows, group_ids, centers, gt_data, modules)
        validate_raw(frame, [seed], params, rows, centers, parsed.sim_len)
        frame.to_csv(seed_path, index=False)
        print(f"Seed {seed}: saved validated seed file {seed_path}", flush=True)
        return

    if parsed.reuse_raw:
        raw = pd.read_csv(raw_path)
        validate_raw(raw, parsed.seeds, params, rows, centers, parsed.sim_len)
    else:
        data_handler = modules[0]
        gt_data = data_handler.load_gt_data(args)
        seed_frames: list[pd.DataFrame] = []
        for seed in parsed.seeds:
            seed_path = data_dir / f"raw_seed_{int(seed)}.csv"
            if seed_path.exists() and not parsed.force:
                candidate = pd.read_csv(seed_path)
                try:
                    validate_raw(candidate, [int(seed)], params, rows, centers, parsed.sim_len)
                except Exception as error:
                    print(f"Seed {seed}: existing file rejected ({error}); rerunning.", flush=True)
                else:
                    print(f"Seed {seed}: reusing validated seed file.", flush=True)
                    seed_frames.append(candidate)
                    continue
            frame = run_seed(
                int(seed), args, params, rows, group_ids, centers, gt_data, modules
            )
            validate_raw(frame, [int(seed)], params, rows, centers, parsed.sim_len)
            frame.to_csv(seed_path, index=False)
            seed_frames.append(frame)
        raw = pd.concat(seed_frames, ignore_index=True)
        validate_raw(raw, parsed.seeds, params, rows, centers, parsed.sim_len)
        raw.to_csv(raw_path, index=False)

    per_seed, aggregate, local_aggregate = build_paired_tables(raw)
    summary = build_epsilon_summary(aggregate)
    primary_epsilon = choose_primary_epsilon(summary)

    per_seed.to_csv(data_dir / "per_seed_finite_differences.csv", index=False)
    aggregate.to_csv(data_dir / "aggregate_finite_differences.csv", index=False)
    local_aggregate.to_csv(data_dir / "local_landscape_summary.csv", index=False)
    summary.to_csv(data_dir / "epsilon_summary.csv", index=False)
    make_plots(local_aggregate, aggregate, summary, plot_dir)

    primary = aggregate[np.isclose(aggregate["epsilon"], primary_epsilon)]
    metadata = {
        "analysis": "Archived STRF-gain local finite-difference pilot",
        "snapshot": str(harness.SNAPSHOT_ROOT),
        "source_harness": str(SOURCE_ANALYSIS),
        "cells": CELLS,
        "seeds": [int(seed) for seed in parsed.seeds],
        "trials_per_seed": 10,
        "num_initialization_batches": int(parsed.num_inits),
        "cell_specific_parameter_contexts": int(parsed.num_inits) * len(CELLS),
        "initialization_method": (
            "Separate stratified scrambled Latin-hypercube designs per cell over the archived linear-uniform "
            "initialization ranges, except that gain centers are lower-truncated so the -10% "
            "log probe remains above 0.001. The +10% probe may extend above the initialization "
            "support but remains inside the model's admissible domain."
        ),
        "initialization_seed": int(parsed.init_seed),
        "parameter_ranges": {name: list(bounds) for name, bounds in PARAMETER_RANGES.items()},
        "gain_center_lower_bound_after_probe_guard": float(0.001 * math.exp(float(LOG_EPSILONS.max()))),
        "log_gain_epsilons": [float(value) for value in LOG_EPSILONS],
        "batch_rows": len(rows),
        "sim_len": int(parsed.sim_len),
        "dt_ms": float(args["simulation"]["dt"]),
        "predeclared_primary_log_epsilon": float(primary_epsilon),
        "primary_counts": {
            "contexts": int(len(primary)),
            "green": int((primary["status"] == "green").sum()),
            "red": int((primary["status"] == "red").sum()),
            "gray": int((primary["status"] == "gray").sum()),
        },
        "classification": (
            "Green/red requires the eligibility nominal 95% t interval to exclude zero and the numerical FD "
            "direction to belong to at least one run of three adjacent epsilons whose nominal 95% t intervals "
            "all exclude zero with the same sign. Green means matching signs; red means opposing signs; "
            "all other comparisons are gray. This is a stable-sign rule, not a magnitude-convergence "
            "criterion. Magnitude stability is reported separately when three adjacent resolved FD "
            "intervals share a common intersection. Intervals are exploratory and are not corrected "
            "for multiple testing."
        ),
        "common_random_numbers": (
            "Torch onset, offset, and output-gate draws are identical across all gain probes within a "
            "center but independent across initialization groups. Archived numpy spontaneous input has "
            "no batch axis and is therefore shared across centers within a seed."
        ),
        "interpretation": (
            "The central finite difference estimates the slope of expected sampled PSTH SSE under paired "
            "Monte Carlo draws. It is not the ordinary derivative of one discontinuous hard-spike trajectory."
        ),
        "elapsed_seconds_this_invocation": time.perf_counter() - start,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    with (data_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nEpsilon summary:")
    print(
        summary[
            [
                "epsilon_percent_log",
                "fd_resolved",
                "stable_sign_window",
                "magnitude_stable_window",
                "qualified",
                "green",
                "red",
                "gray",
                "raw_alignment_fraction",
                "zero_difference_fraction",
            ]
        ].to_string(index=False)
    )
    print(f"\nPredeclared primary epsilon: {100 * primary_epsilon:g}% log gain")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
