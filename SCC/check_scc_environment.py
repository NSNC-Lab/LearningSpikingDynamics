"""Fast SCC environment and project smoke test; does not run an epoch."""

from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.signal import ShortTimeFFT
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true", help="Permit a local CPU-only smoke test.")
    options = parser.parse_args()

    config_path = ROOT / "simulation_config.yaml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = [args["paths"][key] for key in ("stimuli", "data", "spontaneous_activity")]
    missing = [str(resolve_project_path(path)) for path in required if not resolve_project_path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing configured input path(s):\n  " + "\n  ".join(missing))
    target = resolve_project_path(args["paths"]["stimuli"]) / "200k_target1.wav"
    if not target.exists():
        raise FileNotFoundError(f"Missing stimulus: {target}")

    mat_config = args.get("parameter_initialization", {}).get("from_mat", {})
    if mat_config.get("enabled") and not resolve_project_path(mat_config["path"]).exists():
        raise FileNotFoundError(
            "Checkpoint loading is enabled, but the SCC cannot find this path: "
            f"{mat_config['path']}\nUse a Linux /projectnb/... path in simulation_config.yaml."
        )

    if not torch.cuda.is_available() and not options.allow_cpu:
        raise RuntimeError("PyTorch cannot see an assigned CUDA GPU. Submit with qsub/qrsh and -l gpus=1.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Python:       {platform.python_version()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Torch CUDA:   {torch.version.cuda}")
    print(f"NumPy:        {np.__version__}")
    print(f"SciPy:        {scipy.__version__}")
    print(f"Matplotlib:   {matplotlib.__version__}")
    print(f"CUDA visible: {os.getenv('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Test device:  {device}")

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print(f"GPU:          {properties.name}")
        print(f"Capability:   {properties.major}.{properties.minor}")
        print(f"GPU memory:   {free_bytes/2**30:.1f}/{total_bytes/2**30:.1f} GiB free")
        x = torch.randn((1024, 1024), device=device)
        torch.mm(x, x)
        torch.cuda.synchronize()
        del x

    # Construct a tiny network to catch package/import/device incompatibilities
    # without allocating the full configured batch or starting optimization.
    from Simulation import Architecture_Declaration, Parameter_initialization

    smoke_args = copy.deepcopy(args)
    smoke_args["simulation"].update(batch_size=1, cell_targets=[args["simulation"]["cell_targets"][0]],
                                    epochs=1, sim_len=101, device=str(device))
    smoke_args.setdefault("parameter_initialization", {}).setdefault("from_mat", {})["enabled"] = False
    params = Parameter_initialization.init_params(smoke_args)
    states = Architecture_Declaration.build_network(smoke_args, device, params["params"])
    assert states["neurons"]["Learnable"]["STRF_gain"].device == device
    del states
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("SCC smoke test passed. Starting the configured run.")


if __name__ == "__main__":
    main()
