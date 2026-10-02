"""Matched frozen-parameter E-prop/PyTorch BPTT experiment (no optimizer steps)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import torch
import yaml

from autograd_model import Model, PARAMETERS, cosine


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array(tensor):
    return tensor.detach().cpu().numpy()


def gradient_vector(states):
    n, s = states["neurons"]["Learnable"], states["synapses"]["Learnable"]
    values = [n["STRF_gain_grad"], n["STRF_alpha_grad"], n["ron"]["output_ad_grad"]]
    values += [s[k.removesuffix("_gSYN")]["gSYN_grad"] for k in PARAMETERS[3:8]]
    values += [n["ron"][k + "_grad"] for k in PARAMETERS[8:]]
    return torch.stack(values, -1)[:, 0]


def prepare(root, cli, config, architecture):
    from Simulation import Parameter_initialization
    from Pre_Processing import preprocess_handler
    from Data import data_handler

    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    params = Parameter_initialization.init_params(config)["params"]
    device = torch.device(config["simulation"]["device"])
    template = architecture.build_network(config, device, params)
    with torch.no_grad():
        rates = preprocess_handler.get_pre_cortical_handler(config).create_input_fr(config, template, device)
    steps = config["simulation"]["sim_len"]
    if min(value.shape[0] for value in rates.values()) < steps:
        raise ValueError("Requested duration exceeds the available preprocessed stimulus.")
    # Match the float32 voltage engine, retaining the original rate values for sampling.
    inputs = {}
    for index, name in enumerate(("onset", "offset")):
        gen = torch.Generator(device=device).manual_seed(cli.seed + 100 + index)
        probability = rates[name + "_rate"][:steps].permute(1, 2, 0)[:, None]
        inputs[name + "_spks"] = (torch.rand((cli.batch_size, 10, 1, steps), generator=gen, device=device)
                                   < probability * config["simulation"]["dt"] / 1000).float()
    rates = {k: v[:steps].detach().float() for k, v in rates.items()}
    spontaneous = np.load(root / config["paths"]["spontaneous_activity"])
    rate = float(np.mean(spontaneous[cli.cell - 1, :4]))
    if not np.isfinite(rate) or rate < 0:
        raise ValueError(f"Invalid spontaneous firing rate: {rate}")
    gen = torch.Generator(device=device).manual_seed(cli.seed + 200)
    noise = (torch.rand((cli.batch_size, 10, 1, steps), generator=gen, device=device)
             < rate * config["simulation"]["dt"] / 1000).float()
    # Existing loader assumes the full acquisition duration; only slice AFTER loading.
    full_config = copy.deepcopy(config)
    full_config["simulation"]["sim_len"] = cli.acquisition_steps
    target = data_handler.load_gt_data(full_config)["psth_holder"][:, :steps // config["simulation"]["PSTH_granularity"]]
    return params, template, inputs, rates, noise, target


def reference(config, params, architecture, legacy, inputs, rates, noise, target, seed):
    """Run original July forward/E-prop code, one independent noise stream per member."""
    sim = config["simulation"]
    batch, steps, width = sim["batch_size"], sim["sim_len"], sim["PSTH_granularity"]
    device = torch.device(sim["device"])
    gradients, rasters, tapes, voltage = [], [], [], []
    for b in range(batch):
        cfg = copy.deepcopy(config)
        cfg["simulation"]["batch_size"] = 1
        state = architecture.build_network(cfg, device, {k: v[b:b + 1].copy() for k, v in params.items()})
        spikes = {"onset_offset_spks": {k: v[b:b + 1] for k, v in inputs.items()}, "noise_spks": noise[b]}
        member_rates = {k: v[:, b:b + 1] for k, v in rates.items()}
        torch.manual_seed(seed + 1000 + b)
        emitted, uniforms, vs = [], [], []
        previous = gradient_vector(state).clone()
        with torch.no_grad():
            for t in range(steps):
                legacy["ode"].run_odes(cfg, state, spikes, t)
                # Replay the one uniform draw made by legacy conditionals, without patching it.
                rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
                uniforms.append(torch.rand((1, 10, 1), device=device))
                if device.type == "cuda":
                    torch.cuda.set_rng_state(rng, device)
                else:
                    torch.set_rng_state(rng)
                legacy["conditional"].run_conditionals(cfg, state, t)
                legacy["eligibility"].update_eligibility(cfg, state, member_rates, t)
                vs.append(torch.stack([state["neurons"]["Dynamic"][k]["V"][..., -1].clone()
                                       for k in ("on", "off", "sonoff", "ron")], 0))
                if (t + 1) % width == 0:
                    count = state["neurons"]["Dynamic"]["ron"]["spikes_holder"][..., t + 1 - width:t + 1].sum((1, 3))
                    # Same shifted count residual as the root loss handler, for BOTH methods.
                    learning_signal = 2 * (count - target[:, (t + 1) // width - 1][None] - 0.5)
                    legacy["loss"].update_grad(state, learning_signal)
                    current = gradient_vector(state).clone()
                    emitted.append(current - previous)
                    previous = current
        gradients.append(torch.stack(emitted, 0))
        tapes.append(torch.stack(uniforms, 0))
        voltage.append(torch.stack(vs, 0))
        rasters.append(state["neurons"]["Dynamic"]["ron"]["spikes_holder"].bool())
        print(f"E-prop member {b + 1}/{batch}: {int(rasters[-1].sum())} spikes", flush=True)
    return (torch.cat(gradients, 1), torch.cat(rasters, 0), torch.cat(tapes, 1), torch.cat(voltage, 2))


def autograd_run(config, template, params, inputs, rates, noise, uniforms, target, reference_v, reset_gradient):
    sim = config["simulation"]
    p = {k: torch.tensor(params[k], dtype=torch.float32, device=sim["device"], requires_grad=True) for k in PARAMETERS}
    model = Model(template, p, sim["dt"], reset_gradient)
    leaves = tuple(p.values())
    rows, losses, rasters, bin_spikes = [], [], [], []
    max_voltage_error = 0.0
    for t in range(sim["sim_len"]):
        z = model.step(t, inputs, rates, noise, uniforms[t])
        bin_spikes.append(z)
        rasters.append(z.detach().bool())
        with torch.no_grad():
            actual = torch.stack([model.n[k]["V"] for k in ("on", "off", "sonoff", "ron")])
            max_voltage_error = max(max_voltage_error, float((actual - reference_v[t]).abs().max()))
        if (t + 1) % sim["PSTH_granularity"] == 0:
            count = torch.stack(bin_spikes).sum((0, 2))[:, 0]
            bin_index = (t + 1) // sim["PSTH_granularity"] - 1
            loss = ((count - target[0, bin_index] - 0.5) ** 2).sum()
            grads = torch.autograd.grad(loss, leaves, retain_graph=True)
            rows.append(torch.stack(grads, -1)[:, 0].detach())
            losses.append(loss)
            bin_spikes.clear()
            if len(rows) % 10 == 0:
                print(f"BPTT emission {len(rows)} at {(t + 1) * sim['dt']:g} ms", flush=True)
    total = torch.stack(torch.autograd.grad(torch.stack(losses).sum(), leaves), -1)[:, 0]
    emitted = torch.stack(rows)
    if not torch.isfinite(emitted).all():
        raise RuntimeError("Non-finite BPTT gradients; no alignment claim is valid.")
    torch.testing.assert_close(emitted.sum(0), total, rtol=3e-4, atol=3e-4)
    sum_error = (emitted.sum(0) - total).abs()
    relative_error = sum_error / total.abs().clamp_min(1e-6)
    return (emitted, torch.stack(rasters, -1), max_voltage_error,
            float(sum_error.max()), float(relative_error.max()))


def plot_results(out, result, cell, dt):
    time_ms = result["emission_time_ms"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, key, title in zip(axes.flat,
            ("cosine_bin_psth8", "cosine_bin_gsyn", "cosine_cumulative_psth8", "cosine_cumulative_gsyn"),
            ("Per-bin: 8 PSTH parameters", "Per-bin: 5 gSYNs", "Cumulative: 8 PSTH parameters", "Cumulative: 5 gSYNs")):
        for b in range(result[key].shape[1]):
            ax.plot(time_ms, result[key][:, b], lw=1, alpha=.75, label=f"Batch {b + 1}")
        ax.axhline(0, color="gray", lw=.7)
        ax.set(title=title, ylim=(-1.05, 1.05), xlabel="Time (ms)", ylabel="Cosine similarity")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Cell {cell}: fixed parameters; full-history BPTT")
    fig.savefig(out / "gradient_alignment.png", dpi=160)
    plt.close(fig)
    for b in range(result["raster_eprop"].shape[0]):
        fig, axes = plt.subplots(2, 1, figsize=(11, 4), sharex=True, constrained_layout=True)
        for ax, key, label in zip(axes, ("raster_eprop", "raster_bptt"), ("E-prop", "BPTT")):
            trial, t = np.where(result[key][b, :, 0])
            ax.scatter(t * dt, trial + 1, marker="|", s=14, color="black")
            ax.set(title=f"{label}: cell {cell}, batch {b + 1}, {len(t)} spikes", ylabel="Trial", ylim=(.5, 10.5))
        axes[-1].set_xlabel("Time (ms)")
        fig.savefig(out / f"rasters_batch_{b + 1:03d}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--duration-ms", type=float, default=300, help="Complete 10-ms bins only; default pilot is 300 ms.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--reset-gradient", choices=("detach", "full"), default="detach")
    parser.add_argument("--output-dir", type=Path, default=None)
    cli = parser.parse_args()
    if cli.batch_size < 1 or cli.cell < 1 or cli.duration_ms <= 0:
        parser.error("Batch size, cell, and duration must be positive.")
    root = cli.project_root.resolve()
    sys.path.insert(0, str(root))
    from Simulation import Architecture_Declaration as architecture
    config = yaml.safe_load((root / "simulation_config.yaml").read_text(encoding="utf-8"))
    cli.acquisition_steps = config["simulation"]["sim_len"]
    sim = config["simulation"]
    dt = sim["dt"]
    if not np.isclose(dt, 0.1):
        parser.error("The existing ground-truth loader assumes dt=0.1 ms; this experiment preserves that assumption.")
    width = round(10 / dt)
    steps = round(cli.duration_ms / dt)
    if not np.isclose(width * dt, 10) or steps % width or not np.isclose(steps * dt, cli.duration_ms):
        parser.error("Duration must consist of whole 10-ms bins, and dt must divide 10 ms.")
    if steps > cli.acquisition_steps:
        parser.error("Duration exceeds the configured acquisition length.")
    device = "cuda" if cli.device == "auto" and torch.cuda.is_available() else ("cpu" if cli.device == "auto" else cli.device)
    sim.update(batch_size=cli.batch_size, cell_targets=[cli.cell], device=device,
               sim_len=steps, epochs=1, PSTH_granularity=width)
    config.setdefault("parameter_initialization", {}).setdefault("from_mat", {})["enabled"] = False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = cli.output_dir or Path(__file__).resolve().parent / "results" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    legacy_root = root / "Archive/july1_local_eligibility_backup_2026-07-15/Simulation"
    legacy = {key: load_file(legacy_root / filename, "comparison_" + key)
              for key, filename in (("ode", "ode_handler.py"), ("conditional", "conditional_handler.py"),
                                    ("eligibility", "Eligibility_handler.py"), ("loss", "Loss_handler.py"))}
    params, template, inputs, rates, noise, target = prepare(root, cli, config, architecture)
    frozen = {"parameter_names": np.array(PARAMETERS), "initial_parameters": np.stack([params[k] for k in PARAMETERS], -1),
              "noise_spikes": array(noise).astype(bool), "target_counts": array(target)}
    frozen.update({k: array(v).astype(bool) for k, v in inputs.items()})
    frozen.update({k: array(v) for k, v in rates.items()})
    e, raster_e, uniforms, reference_v = reference(config, params, architecture, legacy, inputs, rates, noise, target, cli.seed)
    frozen["output_uniforms"] = array(uniforms)
    np.savez_compressed(out / "frozen_inputs.npz", **frozen)
    b, raster_b, voltage_error, sum_error, relative_sum_error = autograd_run(config, template, params, inputs, rates, noise, uniforms, target, reference_v, cli.reset_gradient)
    e, b = array(e), array(b)
    if not np.isfinite(e).all():
        raise RuntimeError("Non-finite E-prop emissions; no alignment claim is valid.")
    result = {"parameter_names": np.array(PARAMETERS), "gradient_eprop": e, "gradient_bptt": b,
              "gradient_cumulative_eprop": np.cumsum(e, 0), "gradient_cumulative_bptt": np.cumsum(b, 0),
              "raster_eprop": array(raster_e), "raster_bptt": array(raster_b),
              "emission_time_ms": np.arange(1, len(e) + 1) * 10,
              "initial_parameters": frozen["initial_parameters"], "target_counts": array(target)}
    for name, idx in (("psth8", slice(0, 8)), ("gsyn", slice(3, 8))):
        result["cosine_bin_" + name] = cosine(e[..., idx], b[..., idx])
        result["cosine_cumulative_" + name] = cosine(np.cumsum(e[..., idx], 0), np.cumsum(b[..., idx], 0))
        result["norm_eprop_" + name] = np.linalg.norm(e[..., idx], axis=-1)
        result["norm_bptt_" + name] = np.linalg.norm(b[..., idx], axis=-1)
    mismatch = int(np.count_nonzero(result["raster_eprop"] != result["raster_bptt"]))
    sources = [root / "simulation_config.yaml", root / "Simulation/conditional_handler.py", root / "Simulation/ode_handler.py",
               root / "Simulation/Architecture_Declaration.py", root / "Pre_Processing/pre_cortical_handler.py"] + list(legacy_root / f for f in ("ode_handler.py", "conditional_handler.py", "Eligibility_handler.py", "Loss_handler.py"))
    metadata = {"cell": cli.cell, "batch_size": cli.batch_size, "seed": cli.seed, "device": device,
                "duration_ms": cli.duration_ms, "dt_ms": dt, "reset_gradient": cli.reset_gradient,
                "torch_version": torch.__version__, "raster_mismatches": mismatch,
                "max_voltage_difference_mV": voltage_error, "bin_sum_absolute_error": sum_error,
                "bin_sum_max_relative_error": relative_sum_error,
                "spikes_per_member": result["raster_eprop"].sum((1, 2, 3)).tolist(),
                "elapsed_seconds": time.perf_counter() - started,
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "objective": "(sum_trials_and_time(spikes)-target_count-0.5)^2, independent per batch",
                "scope": "PSTH only; no CV, rate loss, Adam, updates, or state truncation. E-prop refractory PSTH gradients are zero.",
                "gradient_axes": "emission, batch, parameter"}
    np.savez_compressed(out / "comparison.npz", **result)
    savemat(out / "comparison.mat", result, do_compression=True)
    (out / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if mismatch or voltage_error > .002:
        raise RuntimeError(f"Forward validation failed: {mismatch} raster mismatches; voltage error {voltage_error:g}. Saved diagnostics, withheld alignment plots.")
    plot_results(out, result, cli.cell, dt)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Results: {out.resolve()}", flush=True)
    if not all(metadata["spikes_per_member"]):
        print("WARNING: at least one member is silent; inspect rasters before interpreting alignment.", flush=True)


if __name__ == "__main__":
    main()
