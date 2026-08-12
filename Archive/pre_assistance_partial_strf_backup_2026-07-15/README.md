# Pre-assistance partial STRF-sensitivity backup

Created on 2026-07-15 immediately before replacing the reconstructed July 9 pre-assistance runtime with the simpler July 1 Git version.

This directory contains a path-preserving copy of the active Python source, configuration, and VS Code launch file (20 files), plus `pre_assistance_vs_b5da538.patch`.

All 20 copied files were SHA-256 compared with the active tree before the July 1 rollback and matched exactly. The patch SHA-256 at creation was `0686715DFE794DD090E1A2E540DF79D1B53F0F70B12BF6B2C9A6156F52498D46`.

This snapshot is the July 9 implementation that still contains the custom `dPSC_*`, `dV_gain`/`dV_alpha`, and `local_pre_derivs` STRF path. Recordings, checkpoints, `.mat` results, plots, and cached bytecode were not duplicated or changed.
