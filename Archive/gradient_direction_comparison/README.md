# E-prop versus full forward-sensitivity direction

This diagnostic runs cell 7 with one batch and ten trials. It uses the July 1
local-eligibility implementation for E-prop and the repository-root recursive
forward-sensitivity implementation for the comparison.

The July E-prop preprocessor is run once. Its parameters, onset spikes, offset
spikes, noise spikes, rates, and rate derivatives are reused verbatim by both
engines. The probabilistic output-spike random stream is reset to the same seed
before each engine. The loss handler and optimizer are not called.

Run from the repository root:

```bash
qsub Archive/gradient_direction_comparison/run_gradient_direction_comparison.qsub
```

For a short CPU smoke test:

```bash
python Archive/gradient_direction_comparison/run_gradient_direction_comparison.py \
  --device cpu --sim-len 500
```

Results are written to `Archive/gradient_direction_comparison/results/`:

- `frozen_inputs_and_parameters.npz`: exact inputs and initial parameters.
- `gradient_direction_comparison.npz`: raw and comparison-ready 12-parameter
  traces, vector magnitudes, cosine similarity/distance, and both rasters.
- `gradient_cosine_similarity.png`: similarity over time for all 12 parameters
  and for the first eight parameters used by the PSTH/rate path.
- `parameter_accumulation_traces.png`: the accumulated trace for every parameter.
- `output_rasters.png`: E-prop and full-sensitivity output rasters.
- `summary.json`: seeds, spike counts, timing, raster equality, and final metrics.

The E-prop raw trace is retained exactly as stored by the July implementation.
For the cosine comparison, its two hidden conductance traces are multiplied by
`Bk`, because July E-prop applies that chain factor inside its loss handler while
the full-sensitivity implementation carries the hidden path recursively before
the loss. Timesteps where either entire vector is zero have undefined (`NaN`)
cosine similarity.

The repository-root `Simulation/conditional_handler.py` was aligned with the
July forward process so that a presynaptic event adds the previously stored
`PSC_q` to `PSC_x` before refreshing `PSC_q`. The corresponding full-sensitivity
event derivatives use the same ordering. The comparison also reuses July's
preprocessed offset rates verbatim, so no independent BPTT preprocessing or
random spike generation can enter this experiment.
