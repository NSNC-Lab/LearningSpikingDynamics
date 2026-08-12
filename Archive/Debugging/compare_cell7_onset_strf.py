from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import wavfile
from scipy.signal import fftconvolve


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_REPO = Path(r"C:\Users\ipboy\Documents\GitHub\MouseSpatialGrid\LearningModels\E-prop")


def best_lag(reference: np.ndarray, candidate: np.ndarray, max_lag: int = 5000) -> tuple[int, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    n = min(reference.size, candidate.size)
    reference = reference[:n] - np.mean(reference[:n])
    candidate = candidate[:n] - np.mean(candidate[:n])

    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = reference[-lag:]
            y = candidate[: n + lag]
        elif lag > 0:
            x = reference[: n - lag]
            y = candidate[lag:]
        else:
            x = reference
            y = candidate

        denom = np.linalg.norm(x) * np.linalg.norm(y)
        corr = float(np.dot(x, y) / denom) if denom else float("nan")
        if corr > best[1]:
            best = (lag, corr)
    return best


def trace_metrics(reference: np.ndarray, candidate: np.ndarray, dt_ms: float) -> dict[str, float | int | None]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    n = min(reference.size, candidate.size)
    reference = reference[:n]
    candidate = candidate[:n]
    error = candidate - reference

    lag, lag_corr = best_lag(reference, candidate)
    denom = np.linalg.norm(reference)
    first_ref = np.flatnonzero(reference > 1e-9)
    first_cand = np.flatnonzero(candidate > 1e-9)

    return {
        "samples": int(n),
        "corr_zero_lag": float(np.corrcoef(reference, candidate)[0, 1]),
        "best_lag_samples_positive_candidate_lags": int(lag),
        "best_lag_ms_positive_candidate_lags": float(lag * dt_ms),
        "best_lag_corr": float(lag_corr),
        "nrmse": float(np.linalg.norm(error) / denom) if denom else float("nan"),
        "mae": float(np.mean(np.abs(error))),
        "reference_peak": float(np.max(reference)),
        "candidate_peak": float(np.max(candidate)),
        "peak_ratio_candidate_over_reference": float(np.max(candidate) / np.max(reference)),
        "reference_peak_index": int(np.argmax(reference)),
        "candidate_peak_index": int(np.argmax(candidate)),
        "reference_area": float(np.sum(reference)),
        "candidate_area": float(np.sum(candidate)),
        "area_ratio_candidate_over_reference": float(np.sum(candidate) / np.sum(reference)),
        "reference_first_nonzero_index": int(first_ref[0]) if first_ref.size else None,
        "candidate_first_nonzero_index": int(first_cand[0]) if first_cand.size else None,
    }


def old_onset_rate(old_repo: Path, gain: float, alpha_ms: float, batch_size: int) -> np.ndarray:
    sys.path.insert(0, str(old_repo))
    cwd = Path.cwd()
    try:
        os.chdir(old_repo)
        old_gen = importlib.import_module("gen_strf_Learnable_STRF")
        old_gen.cp = None
        gen_strf = old_gen.GenSTRF
        rms = old_gen.rms

        config_path = old_repo.parent / "config" / "config.yaml"
        config = yaml.safe_load(open(config_path, "r"))["strf_config"]
        target_path = old_repo.parent / "resampled-stimuli" / "target" / "200k_target1.wav"
        p = np.zeros((8, 1, batch_size), dtype=np.float64)
        p[0, :, :] = gain
        p[1, :, :] = alpha_ms / 1000.0
        p[2, :, :] = 0.001
        p[3:, :, :] = 0.01

        old_args = argparse.Namespace(target_dir=str(target_path.parent), masker_dir="")
        generator = gen_strf(old_args, config, str(target_path), p, batch_size, num_cells=1)

        fs, data = wavfile.read(target_path)
        data = data.astype(np.float64)
        stim_spec, _, _ = generator.STRFspectrogram(data[np.newaxis, :] / rms(data) * config["targetlvl"], fs)
        stim_spec = stim_spec * config["stimGain"]

        drive = np.einsum("tf,f->t", stim_spec, np.asarray(generator.strf["G"], dtype=np.float64))
        temporal_kernel = np.asarray(generator.strf["H"], dtype=np.float64)
        temporal_kernel = np.squeeze(temporal_kernel)
        if temporal_kernel.ndim == 1:
            temporal_kernel = temporal_kernel[:, None]
        temporal_kernel = temporal_kernel * gain
        samples = stim_spec.shape[0]
        if temporal_kernel.ndim == 2:
            response = fftconvolve(drive[:, None], temporal_kernel, mode="full", axes=0)[:samples, 0]
        else:
            response = fftconvolve(drive[:, None, None], temporal_kernel, mode="full", axes=0)[:samples, 0, 0]

        onset = response * config["mean_rate"]
        onset[onset < 0] = 0
        return onset[2500:]
    finally:
        os.chdir(cwd)
        try:
            sys.path.remove(str(old_repo))
        except ValueError:
            pass


def current_onset_rate(gain: float, alpha_ms: float, batch_size: int, cell: int) -> tuple[np.ndarray, dict]:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from Pre_Processing import pre_cortical_handler
    finally:
        try:
            sys.path.remove(str(REPO_ROOT))
        except ValueError:
            pass

    args = yaml.safe_load(open(REPO_ROOT / "simulation_config.yaml", "r"))
    args["simulation"]["device"] = "cpu"
    args["simulation"]["batch_size"] = batch_size
    args["simulation"]["cell_targets"] = [cell]

    states = {
        "neurons": {
            "Learnable": {
                "STRF_gain": torch.full((batch_size, 1), gain, dtype=torch.float32),
                "STRF_alpha": torch.full((batch_size, 1), alpha_ms, dtype=torch.float32),
            }
        }
    }
    rates = pre_cortical_handler.create_input_fr(args, states, torch.device("cpu"))
    onset = rates["onset_rate"].detach().cpu().numpy()
    return onset.reshape(onset.shape[0], -1)[:, 0], args


def save_plot(reference: np.ndarray, candidate: np.ndarray, dt_ms: float, out_path: Path) -> None:
    n = min(reference.size, candidate.size)
    t_ms = np.arange(n) * dt_ms

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t_ms, reference[:n], label="old implementation", linewidth=1.2)
    axes[0].plot(t_ms, candidate[:n], label="current implementation", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("onset rate")
    axes[0].legend(frameon=False)
    axes[0].set_title("Cell 7 onset STRF profile")

    axes[1].plot(t_ms, candidate[:n] - reference[:n], color="black", linewidth=0.8)
    axes[1].axhline(0, color="0.7", linewidth=0.8)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("current - old")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-repo", type=Path, default=DEFAULT_OLD_REPO)
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--alpha-ms", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "Archive" / "Debugging" / "cell7_onset_strf_match")
    args_cli = parser.parse_args()

    args_cli.out_dir.mkdir(parents=True, exist_ok=True)
    reference = old_onset_rate(args_cli.old_repo, args_cli.gain, args_cli.alpha_ms, args_cli.batch_size)
    candidate, current_args = current_onset_rate(args_cli.gain, args_cli.alpha_ms, args_cli.batch_size, args_cli.cell)
    metrics = trace_metrics(reference, candidate, current_args["simulation"]["dt"])

    metrics_path = args_cli.out_dir / "cell7_onset_metrics.json"
    plot_path = args_cli.out_dir / "cell7_onset_overlay.png"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plot(reference, candidate, current_args["simulation"]["dt"], plot_path)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
