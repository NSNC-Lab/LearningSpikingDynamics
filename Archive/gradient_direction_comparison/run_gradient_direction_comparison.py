"""Compare loss-free E-prop and full forward-sensitivity gradient directions.

The experiment runs the July 1 local-eligibility implementation and the current
root implementation with one cell, one batch, identical parameters, identical
input spike trains, and an identical random stream for probabilistic output
spiking.  The loss handler is deliberately not called: each saved row is the
cumulative pre-loss sensitivity available after that timestep.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PARAMETER_NAMES = (
    "STRF_gain",
    "STRF_alpha",
    "output_ad",
    "on_ron_gSYN",
    "off_ron_gSYN",
    "on_sonoff_gSYN",
    "off_sonoff_gSYN",
    "sonoff_ron_gSYN",
    "abs_ref",
    "rel_ref_a",
    "rel_ref_b",
    "rel_ref_c",
)
SYNAPSE_NAMES = (
    "on_ron",
    "off_ron",
    "on_sonoff",
    "off_sonoff",
    "sonoff_ron",
)
PROJECT_PACKAGES = ("Simulation", "Pre_Processing", "Data")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, default=7, help="One-indexed cell ID.")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--sim-len",
        type=int,
        default=None,
        help="Number of timesteps; defaults to the complete configured stimulus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--project-root", type=Path, default=project_root)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def purge_project_modules() -> None:
    for module_name in list(sys.modules):
        if any(
            module_name == package or module_name.startswith(package + ".")
            for package in PROJECT_PACKAGES
        ):
            del sys.modules[module_name]


def load_implementation(source_root: Path) -> dict[str, object]:
    purge_project_modules()
    sys.path.insert(0, str(source_root))
    try:
        modules = {
            "architecture": importlib.import_module("Simulation.Architecture_Declaration"),
            "conditional": importlib.import_module("Simulation.conditional_handler"),
            "eligibility": importlib.import_module("Simulation.Eligibility_handler"),
            "ode": importlib.import_module("Simulation.ode_handler"),
            "parameter_initialization": importlib.import_module(
                "Simulation.Parameter_initialization"
            ),
            "preprocess": importlib.import_module("Pre_Processing.preprocess_handler"),
        }
    finally:
        sys.path.pop(0)
    return modules


def load_config(source_root: Path, device: torch.device, cell: int, sim_len: int | None) -> dict:
    with (source_root / "simulation_config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["simulation"]["batch_size"] = 1
    config["simulation"]["cell_targets"] = [cell]
    config["simulation"]["epochs"] = 1
    config["simulation"]["device"] = str(device)
    if sim_len is not None:
        config["simulation"]["sim_len"] = sim_len
    return config


def seed_everything(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def clone_parameter_bundle(parameter_bundle: dict) -> dict:
    cloned = {"params": {name: value.copy() for name, value in parameter_bundle["params"].items()}}
    if "lrs" in parameter_bundle:
        cloned["lrs"] = copy.deepcopy(parameter_bundle["lrs"])
    return cloned


def initialize_single_batch_parameters(initializer: object, config: dict) -> dict:
    """Use July's grouped initializer, then retain one deterministic batch."""
    initialization_config = copy.deepcopy(config)
    initialization_config["simulation"]["batch_size"] = max(
        21, config["simulation"]["batch_size"]
    )
    bundle = initializer.init_params(initialization_config)
    single_batch = clone_parameter_bundle(bundle)
    single_batch["params"] = {
        name: value[:1].copy() for name, value in bundle["params"].items()
    }
    return single_batch


def tensors_to_numpy(tree: dict) -> dict:
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result[key] = tensors_to_numpy(value)
        elif isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().numpy().copy()
        else:
            result[key] = np.asarray(value).copy()
    return result


def numpy_to_tensors(tree: dict, device: torch.device) -> dict:
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result[key] = numpy_to_tensors(value, device)
        else:
            result[key] = torch.as_tensor(value, device=device)
    return result


def extract_raw_trace(states: dict) -> np.ndarray:
    learnable = states["neurons"]["Learnable"]
    output = learnable["ron"]
    values = [
        learnable["STRF_gain_accum"],
        learnable["STRF_alpha_accum"],
        output["output_ad_accum"],
    ]
    values.extend(
        states["synapses"]["Learnable"][name]["gSYN_accum"]
        for name in SYNAPSE_NAMES
    )
    values.extend(
        output[name]
        for name in (
            "abs_ref_accum",
            "rel_ref_a_accum",
            "rel_ref_b_accum",
            "rel_ref_c_accum",
        )
    )
    return np.asarray([float(value.detach().cpu().reshape(-1)[0]) for value in values])


def eprop_comparison_trace(states: dict, raw_trace: np.ndarray) -> np.ndarray:
    """Map local hidden eligibility traces into output sensitivity space.

    The July implementation applies Bk to on/off -> sonoff inside update_grad.
    Applying it here makes the two hidden-conductance entries comparable with
    the recursive output sensitivities in the root implementation while still
    preserving the untouched values in eprop_raw_trace.
    """
    trace = raw_trace.copy()
    bk = states["neurons"]["Learnable"].get("Bk")
    if bk is not None:
        bk_value = float(bk.detach().cpu().reshape(-1)[0])
        trace[5:7] *= bk_value
    return trace


def run_engine(
    engine: str,
    source_root: Path,
    config: dict,
    parameters: dict,
    shared_spikes: dict,
    shared_rates: dict,
    device: torch.device,
    forward_seed: int,
    progress_every: int,
) -> dict[str, np.ndarray]:
    modules = load_implementation(source_root)
    states = modules["architecture"].build_network(config, device, parameters["params"])
    spikes = numpy_to_tensors(shared_spikes, device)
    rates = numpy_to_tensors(shared_rates, device)
    sim_len = config["simulation"]["sim_len"]
    raw_trace = np.empty((sim_len, len(PARAMETER_NAMES)), dtype=np.float64)
    comparison_trace = np.empty_like(raw_trace)

    seed_everything(forward_seed, device)
    started = time.perf_counter()
    with torch.no_grad():
        for timestep in range(sim_len):
            if engine == "eprop":
                states = modules["ode"].run_odes(config, states, spikes, timestep)
            else:
                states = modules["ode"].run_odes(config, states, spikes, rates, timestep)
            states = modules["conditional"].run_conditionals(config, states, timestep)
            states = modules["eligibility"].update_eligibility(
                config, states, rates, timestep
            )
            raw_trace[timestep] = extract_raw_trace(states)
            comparison_trace[timestep] = (
                eprop_comparison_trace(states, raw_trace[timestep])
                if engine == "eprop"
                else raw_trace[timestep]
            )
            if progress_every and (timestep + 1) % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(f"{engine}: {timestep + 1}/{sim_len} timesteps ({elapsed:.1f} s)", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    raster = (
        states["neurons"]["Dynamic"]["ron"]["spikes_holder"]
        .detach()
        .cpu()
        .numpy()[0, :, 0, :]
        .astype(np.uint8)
    )
    return {
        "raw_trace": raw_trace,
        "comparison_trace": comparison_trace,
        "raster": raster,
        "elapsed_seconds": np.asarray(elapsed),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    similarity = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).eps
    similarity[valid] = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return np.clip(similarity, -1.0, 1.0)


def save_shared_inputs(
    output_dir: Path,
    parameters: dict,
    spikes: dict,
    rates: dict,
) -> None:
    payload = {}
    for name, value in parameters["params"].items():
        payload[f"parameter__{name}"] = value
    for group, tree in (("spikes", spikes), ("rates", rates)):
        for name, value in tree.items():
            if isinstance(value, dict):
                for child_name, child_value in value.items():
                    payload[f"{group}__{name}__{child_name}"] = child_value
            else:
                payload[f"{group}__{name}"] = value
    np.savez_compressed(output_dir / "frozen_inputs_and_parameters.npz", **payload)


def plot_similarity(
    output_dir: Path,
    time_seconds: np.ndarray,
    cosine_all: np.ndarray,
    cosine_optimized: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(time_seconds, cosine_all, color="#176B87", linewidth=1.2, label="All 12 parameters")
    ax.plot(
        time_seconds,
        cosine_optimized,
        color="#C84B31",
        linewidth=1.0,
        alpha=0.85,
        label="First 8 parameters",
    )
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.set(xlabel="Time (s)", ylabel="Cosine similarity", ylim=(-1.05, 1.05))
    ax.set_title("Accumulated loss-free sensitivity direction: E-prop vs full sensitivity")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "gradient_cosine_similarity.png", dpi=180)
    plt.close(fig)


def plot_parameter_traces(
    output_dir: Path,
    time_seconds: np.ndarray,
    eprop_trace: np.ndarray,
    bptt_trace: np.ndarray,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True, constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes.flat, PARAMETER_NAMES)):
        axis.plot(time_seconds, eprop_trace[:, index], label="E-prop", linewidth=0.9)
        axis.plot(time_seconds, bptt_trace[:, index], label="Full sensitivity", linewidth=0.9)
        axis.set_title(name.replace("_", " "), fontsize=10)
        axis.grid(alpha=0.18)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    fig.suptitle("Cumulative pre-loss parameter sensitivities")
    fig.savefig(output_dir / "parameter_accumulation_traces.png", dpi=180)
    plt.close(fig)


def plot_rasters(
    output_dir: Path,
    eprop_raster: np.ndarray,
    bptt_raster: np.ndarray,
    dt_ms: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, constrained_layout=True)
    for axis, raster, title in zip(
        axes,
        (eprop_raster, bptt_raster),
        ("E-prop output", "Full-sensitivity output"),
    ):
        for trial in range(raster.shape[0]):
            spike_times = np.flatnonzero(raster[trial]) * dt_ms / 1000.0
            axis.vlines(spike_times, trial + 0.6, trial + 1.4, color="black", linewidth=0.7)
        axis.set_ylim(0.5, raster.shape[0] + 0.5)
        axis.set_yticks(range(1, raster.shape[0] + 1))
        axis.set_ylabel("Trial")
        axis.set_title(f"{title} ({int(raster.sum())} spikes)")
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(output_dir / "output_rasters.png", dpi=180)
    plt.close(fig)


def main() -> int:
    options = parse_args()
    project_root = options.project_root.resolve()
    eprop_root = project_root / "Archive" / "july1_local_eligibility_backup_2026-07-15"
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(options.device)

    eprop_modules = load_implementation(eprop_root)
    eprop_config = load_config(eprop_root, device, options.cell, options.sim_len)
    sim_len = eprop_config["simulation"]["sim_len"]
    root_config = load_config(project_root, device, options.cell, sim_len)
    root_config["simulation"]["PSTH_granularity"] = eprop_config["simulation"][
        "PSTH_granularity"
    ]

    seed_everything(options.seed, device)
    parameters = initialize_single_batch_parameters(
        eprop_modules["parameter_initialization"], eprop_config
    )
    with torch.no_grad():
        initial_states = eprop_modules["architecture"].build_network(
            eprop_config, device, parameters["params"]
        )
        preprocessed = eprop_modules["preprocess"].preprocess(
            eprop_config, initial_states, device
        )
    shared_spikes = tensors_to_numpy(preprocessed["spks"])
    shared_rates = tensors_to_numpy(preprocessed["onset_offset_rates"])
    save_shared_inputs(output_dir, parameters, shared_spikes, shared_rates)
    del initial_states, preprocessed, eprop_modules
    if device.type == "cuda":
        torch.cuda.empty_cache()

    forward_seed = options.seed + 1
    print(f"Running cell {options.cell}, one batch, {sim_len} timesteps on {device}.")
    eprop_result = run_engine(
        "eprop",
        eprop_root,
        eprop_config,
        parameters,
        shared_spikes,
        shared_rates,
        device,
        forward_seed,
        options.progress_every,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bptt_result = run_engine(
        "bptt",
        project_root,
        root_config,
        parameters,
        shared_spikes,
        shared_rates,
        device,
        forward_seed,
        options.progress_every,
    )

    eprop_trace = eprop_result["comparison_trace"]
    bptt_trace = bptt_result["comparison_trace"]
    cosine_all = cosine_similarity(eprop_trace, bptt_trace)
    cosine_optimized = cosine_similarity(eprop_trace[:, :8], bptt_trace[:, :8])
    raster_equal = np.array_equal(eprop_result["raster"], bptt_result["raster"])
    raster_difference_count = int(
        np.count_nonzero(eprop_result["raster"] != bptt_result["raster"])
    )
    dt_ms = float(eprop_config["simulation"]["dt"])
    time_seconds = np.arange(sim_len) * dt_ms / 1000.0

    np.savez_compressed(
        output_dir / "gradient_direction_comparison.npz",
        parameter_names=np.asarray(PARAMETER_NAMES),
        time_seconds=time_seconds,
        eprop_raw_trace=eprop_result["raw_trace"],
        eprop_comparison_trace=eprop_trace,
        bptt_raw_trace=bptt_result["raw_trace"],
        bptt_comparison_trace=bptt_trace,
        eprop_magnitude=np.linalg.norm(eprop_trace, axis=1),
        bptt_magnitude=np.linalg.norm(bptt_trace, axis=1),
        cosine_similarity=cosine_all,
        cosine_distance=1.0 - cosine_all,
        cosine_similarity_first_8=cosine_optimized,
        eprop_raster=eprop_result["raster"],
        bptt_raster=bptt_result["raster"],
    )
    plot_similarity(output_dir, time_seconds, cosine_all, cosine_optimized)
    plot_parameter_traces(output_dir, time_seconds, eprop_trace, bptt_trace)
    plot_rasters(
        output_dir,
        eprop_result["raster"],
        bptt_result["raster"],
        dt_ms,
    )

    finite = np.isfinite(cosine_all)
    summary = {
        "cell": options.cell,
        "batch_size": 1,
        "trials": int(eprop_result["raster"].shape[0]),
        "timesteps": sim_len,
        "dt_ms": dt_ms,
        "device": str(device),
        "seed": options.seed,
        "forward_seed": forward_seed,
        "eprop_spikes": int(eprop_result["raster"].sum()),
        "bptt_spikes": int(bptt_result["raster"].sum()),
        "eprop_firing_rate_hz": float(
            eprop_result["raster"].sum()
            / (eprop_result["raster"].shape[0] * sim_len * dt_ms / 1000.0)
        ),
        "rasters_identical": bool(raster_equal),
        "raster_difference_count": raster_difference_count,
        "finite_cosine_timesteps": int(finite.sum()),
        "final_cosine_similarity": float(cosine_all[finite][-1]) if finite.any() else None,
        "mean_cosine_similarity": float(np.nanmean(cosine_all)) if finite.any() else None,
        "final_cosine_similarity_first_8": float(cosine_optimized[np.isfinite(cosine_optimized)][-1])
        if np.isfinite(cosine_optimized).any()
        else None,
        "mean_cosine_similarity_first_8": float(np.nanmean(cosine_optimized))
        if np.isfinite(cosine_optimized).any()
        else None,
        "final_matching_nonzero_signs": int(
            np.sum(
                (np.sign(eprop_trace[-1]) == np.sign(bptt_trace[-1]))
                & (eprop_trace[-1] != 0)
                & (bptt_trace[-1] != 0)
            )
        ),
        "final_opposing_nonzero_signs": int(
            np.sum(np.sign(eprop_trace[-1]) * np.sign(bptt_trace[-1]) == -1)
        ),
        "eprop_elapsed_seconds": float(eprop_result["elapsed_seconds"]),
        "bptt_elapsed_seconds": float(bptt_result["elapsed_seconds"]),
        "shared_offset_definition": "July E-prop preprocessed rates and spikes reused verbatim",
        "loss_handler_called": False,
        "eprop_hidden_chain_adjustment": "Bk applied to on/off -> sonoff for cosine comparison",
        "interpretation_note": "Raw cosine depends on parameter units and is dominated here by the full-sensitivity STRF terms.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if not raster_equal:
        print(
            "WARNING: Output rasters differ. Treat the gradient comparison as diagnostic "
            "until the remaining forward-process mismatch is resolved.",
            file=sys.stderr,
        )
        return 2
    if summary["eprop_spikes"] == 0:
        print(
            "WARNING: The frozen run produced no output spikes; choose another seed or "
            "provide a trained initialization before interpreting the cosine trace.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
