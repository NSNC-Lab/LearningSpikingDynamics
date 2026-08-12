from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_difference_strf_eligibility import (  # noqa: E402
    finite_metrics,
    make_test_args,
    perturb_params,
    select_params_from_config,
)
from search_strf_eligibility_variants import run_forward_bins  # noqa: E402


PARAMS = {
    "STRF_gain": "Strf_gain",
    "STRF_alpha": "Strf_alpha",
}


def fd_for_scale(
    args: dict[str, Any],
    params: dict[str, np.ndarray],
    param_name: str,
    seed: int,
    epsilon_scale: float,
) -> dict[str, Any]:
    base_value = float(params[PARAMS[param_name]][0, 0])
    floor = 5e-5 if param_name == "STRF_gain" else 0.25
    epsilon = max(abs(base_value) * epsilon_scale, floor)
    plus = run_forward_bins(args, perturb_params(params, param_name, epsilon), seed)
    minus = run_forward_bins(args, perturb_params(params, param_name, -epsilon), seed)
    fd = (plus - minus) / (2.0 * epsilon)
    return {
        "epsilon": epsilon,
        "fd": fd,
        "nonzero_bins": int(np.count_nonzero(fd)),
        "rms": float(np.sqrt(np.mean(fd**2))),
        "sum": float(np.sum(fd)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose finite-difference STRF target stability.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--cell", type=int, default=7)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--sim-len", type=int, default=3001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    parser.add_argument("--epsilon-scales", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.1])
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Archive" / "Debugging" / "strf_fd_target_stability.json",
    )
    args_cli = parser.parse_args()

    with args_cli.config.open("r", encoding="utf-8") as handle:
        base_args = yaml.safe_load(handle)

    test_args = make_test_args(
        base_args,
        cell_target=args_cli.cell,
        batch_size=1,
        sim_len=args_cli.sim_len,
        device=args_cli.device,
    )
    params = select_params_from_config(base_args, test_args, args_cli.cell, args_cli.batch)

    raw: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    for param_name in PARAMS:
        entries = []
        for seed in args_cli.seeds:
            for scale in args_cli.epsilon_scales:
                result = fd_for_scale(test_args, params, param_name, seed, scale)
                entries.append(
                    {
                        "seed": seed,
                        "epsilon_scale": scale,
                        "epsilon": result["epsilon"],
                        "nonzero_bins": result["nonzero_bins"],
                        "rms": result["rms"],
                        "sum": result["sum"],
                        "fd": result["fd"].tolist(),
                    }
                )
        raw[param_name] = {"entries": entries}

        comparisons = []
        for i, left in enumerate(entries):
            for right in entries[i + 1 :]:
                comparisons.append(
                    {
                        "left_seed": left["seed"],
                        "left_epsilon_scale": left["epsilon_scale"],
                        "right_seed": right["seed"],
                        "right_epsilon_scale": right["epsilon_scale"],
                        "metrics": finite_metrics(np.asarray(left["fd"]), np.asarray(right["fd"])),
                    }
                )
        correlations = [item["metrics"]["correlation"] for item in comparisons]
        sign_agreements = [item["metrics"]["sign_agreement"] for item in comparisons]
        summary[param_name] = {
            "mean_pairwise_correlation": float(np.nanmean(correlations)),
            "median_pairwise_correlation": float(np.nanmedian(correlations)),
            "mean_pairwise_sign_agreement": float(np.nanmean(sign_agreements)),
            "median_pairwise_sign_agreement": float(np.nanmedian(sign_agreements)),
            "comparisons": comparisons,
        }

    output = {
        "cell": args_cli.cell,
        "batch": args_cli.batch,
        "sim_len": args_cli.sim_len,
        "seeds": args_cli.seeds,
        "epsilon_scales": args_cli.epsilon_scales,
        "summary": summary,
        "raw": raw,
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args_cli.output}")


if __name__ == "__main__":
    main()
