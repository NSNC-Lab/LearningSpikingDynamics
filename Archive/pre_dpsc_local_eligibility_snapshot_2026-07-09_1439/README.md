# Pre-dPSC local-eligibility snapshot

This is the reconstructed project state immediately before the first `dPSC_*` work on 2026-07-09 at 14:40 EDT.

The runtime files are the latest VS Code Local History entries at or before 14:39:59. In particular:

- `Pre_Processing/pre_cortical_handler.py` is `Jds6.py` from 2026-07-08 14:20:10. STRF gain is multiplied into `H` before convolution, and separate local gain and alpha convolution derivatives are returned.
- `Simulation/Eligibility_handler.py` is `OjXS.py` from 2026-07-09 14:05:24. It consumes those local derivatives without `dPSC_*`, `dV_*`, or `local_pre_derivs` running state.
- The required pre-existing sidecars `pre_cortical_handler_legacy_match.py` and `initialize_from_mat.py` are included.

One narrowly reconstructed fix is included in `Architecture_Declaration.py`: the eight tracker dimensions use `*params[...].shape`. The raw last pre-14:40 file had malformed nested tuple dimensions and could not build the network. Those tracker-shape fixes were saved seconds later in the same edit that first added `dPSC_x`; the `dPSC_x` declaration has been removed here.

Verification before activation: all Python files parsed, YAML parsed, no sensitivity markers were present, and a CPU `build_network` smoke test produced tracker shape `(2, 3, 5)` with zero `dPSC_*`/`dV_*` state keys.
