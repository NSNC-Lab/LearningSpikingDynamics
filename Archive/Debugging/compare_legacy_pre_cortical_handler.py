from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import wavfile
from scipy.signal import fftconvolve


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_REPO = Path(r"C:\Users\ipboy\Documents\GitHub\MouseSpatialGrid\LearningModels\E-prop")


def _rms(data):
    return np.sqrt(np.mean(data ** 2))


def _as_delay_batch(value, batch_size):
    arr = np.squeeze(np.asarray(value, dtype=np.float64))
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] != batch_size:
        arr = np.broadcast_to(arr[:, :1], (arr.shape[0], batch_size))
    return arr


def _old_reference(old_repo, gain, alpha_ms, batch_size):
    sys.path.insert(0, str(old_repo))
    cwd = Path.cwd()
    try:
        os.chdir(old_repo)
        old_gen = importlib.import_module("gen_strf_Learnable_STRF")
        old_gen.cp = None
        config = yaml.safe_load(open(old_repo.parent / "config" / "config.yaml", "r"))["strf_config"]
        target_path = old_repo.parent / "resampled-stimuli" / "target" / "200k_target1.wav"

        p = np.zeros((8, 1, batch_size), dtype=np.float64)
        p[0, :, :] = gain
        p[1, :, :] = alpha_ms / 1000.0
        old_args = argparse.Namespace(target_dir=str(target_path.parent), masker_dir="")
        generator = old_gen.GenSTRF(old_args, config, str(target_path), p, batch_size, num_cells=1)

        fs, data = wavfile.read(target_path)
        data = data.astype(np.float64)
        stim_spec, _, _ = generator.STRFspectrogram(data[np.newaxis, :] / _rms(data) * config["targetlvl"], fs)
        stim_spec = stim_spec * config["stimGain"]

        drive = np.einsum("tf,f->t", stim_spec, np.asarray(generator.strf["G"], dtype=np.float64))
        h = _as_delay_batch(generator.strf["H"], batch_size) * gain
        h_alpha_seconds = _as_delay_batch(generator.strf["H2"], batch_size) * gain
        samples = stim_spec.shape[0]

        frate = fftconvolve(drive[:, None], h, mode="full", axes=0)[:samples] * config["mean_rate"]
        frate_alpha_seconds = (
            fftconvolve(drive[:, None], h_alpha_seconds, mode="full", axes=0)[:samples] * config["mean_rate"]
        )

        frate = frate[2500:, :, None]
        frate_alpha_seconds = frate_alpha_seconds[2500:, :, None]

        offset_rate = -frate + np.max(frate, axis=0, keepdims=True) * 0.75
        offset_alpha_seconds = -frate_alpha_seconds + np.max(frate_alpha_seconds, axis=0, keepdims=True) * 0.3
        for batch_index in range(batch_size):
            nonpositive = np.where(offset_rate[:, batch_index, 0] <= 0)[0]
            if nonpositive.size:
                first_nonpositive = nonpositive[0]
                offset_rate[:first_nonpositive, batch_index, 0] = 0
                offset_alpha_seconds[:first_nonpositive, batch_index, 0] = 0
        offset_rate[offset_rate < 0] = 0
        offset_alpha_seconds[offset_alpha_seconds < 0] = 0

        onset_rate = frate.copy()
        onset_alpha_seconds = frate_alpha_seconds.copy()
        onset_rate[onset_rate < 0] = 0
        onset_alpha_seconds[onset_alpha_seconds < 0] = 0

        return {
            "onset_rate": onset_rate,
            "offset_rate": offset_rate,
            "onset_rate_deriv_seconds": onset_alpha_seconds,
            "offset_rate_deriv_seconds": offset_alpha_seconds,
        }
    finally:
        os.chdir(cwd)
        try:
            sys.path.remove(str(old_repo))
        except ValueError:
            pass


def _new_output(gain, alpha_ms, batch_size):
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from Pre_Processing import pre_cortical_handler_legacy_match
    finally:
        try:
            sys.path.remove(str(REPO_ROOT))
        except ValueError:
            pass

    args = yaml.safe_load(open(REPO_ROOT / "simulation_config.yaml", "r"))
    args["simulation"]["device"] = "cpu"
    args["simulation"]["batch_size"] = batch_size
    args["simulation"]["cell_targets"] = [7]
    args["strf"].setdefault("legacy_match", {})["alpha_derivative_units"] = "milliseconds"
    states = {
        "neurons": {
            "Learnable": {
                "STRF_gain": torch.full((batch_size, 1), gain, dtype=torch.float32),
                "STRF_alpha": torch.full((batch_size, 1), alpha_ms, dtype=torch.float32),
            }
        }
    }
    result = pre_cortical_handler_legacy_match.create_input_fr(args, states, torch.device("cpu"))
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def _metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    error = candidate - reference
    denom = np.linalg.norm(reference)
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "mae": float(np.mean(np.abs(error))),
        "nrmse": float(np.linalg.norm(error) / denom) if denom else float("nan"),
        "reference_peak": float(np.max(reference)),
        "candidate_peak": float(np.max(candidate)),
        "peak_ratio": float(np.max(candidate) / np.max(reference)) if np.max(reference) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-repo", type=Path, default=DEFAULT_OLD_REPO)
    parser.add_argument("--gain", type=float, default=0.06)
    parser.add_argument("--alpha-ms", type=float, default=8.8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "legacy_pre_cortical_match_metrics.json",
    )
    args = parser.parse_args()

    old = _old_reference(args.old_repo, args.gain, args.alpha_ms, args.batch_size)
    new = _new_output(args.gain, args.alpha_ms, args.batch_size)
    metrics = {
        "onset_rate": _metrics(old["onset_rate"], new["onset_rate"]),
        "offset_rate": _metrics(old["offset_rate"], new["offset_rate"]),
        "onset_rate_deriv_ms": _metrics(old["onset_rate_deriv_seconds"] / 1000.0, new["onset_rate_deriv"]),
        "offset_rate_deriv_ms": _metrics(old["offset_rate_deriv_seconds"] / 1000.0, new["offset_rate_deriv"]),
        "onset_rate_gain_deriv": _metrics(old["onset_rate"] / args.gain, new["onset_rate_gain_deriv"]),
        "offset_rate_gain_deriv": _metrics(old["offset_rate"] / args.gain, new["offset_rate_gain_deriv"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
