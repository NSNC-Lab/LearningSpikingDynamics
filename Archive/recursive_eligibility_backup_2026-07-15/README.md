# Recursive eligibility implementation backup

Created on 2026-07-15 immediately before restoring the project to its pre-assistance eligibility implementation.

## What this contains

- A path-preserving snapshot of the current Python source, `simulation_config.yaml`, `.vscode/launch.json`, helper modules, and the three recursive-eligibility regression tests (23 files total).
- `current_vs_b5da538.patch`, a binary-capable Git patch containing the tracked source/config differences from commit `b5da538af6741d1695c2a34ca51b117ec7f8148f`.

The snapshot files were SHA-256 compared with the active files before rollback: all 23 copies matched. The patch SHA-256 at creation was `80A213318B2A3D11C2EE604A26D3FACEF4273BFD448D4E9B7B193CC3DA646C07`.

## What was intentionally excluded

Recordings, checkpoints, `.mat` results, plots, cached bytecode, and other generated data were not duplicated. They were left unchanged in the working tree.

## Rollback baseline

The exact pre-assistance state is newer than the July 1 Git commit, so the rollback uses the last VS Code Local History version before the original July 10 assistance request for the affected runtime files. `Main.py` and `Pre_Processing/spiking_handler.py`, which were clean at that point, come from `b5da538`. Existing pre-assistance changes in `.vscode/launch.json` and `Pre_Processing/preprocess_handler.py` are preserved.

The archived source tree itself is the easiest way to recover the recursive implementation: copy the desired path-preserving files back to the repository root.
