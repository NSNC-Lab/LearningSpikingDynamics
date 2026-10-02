# Matched E-prop / autograd experiment

This is a frozen-parameter diagnostic, not a training run. It compares the July
local-eligibility implementation with a functional PyTorch model of the current
root forward dynamics. No existing training code is modified.

## Run

From the repository root, in an environment with PyTorch, NumPy, SciPy,
Matplotlib, and PyYAML:

```bash
python Archive/autograd_eprop_comparison/run_comparison.py --cell 7 --batch-size 3 --duration-ms 300 --device cpu
```

Use `--batch-size X` for X independent candidates, each with ten trials. Parameters,
onset/offset spikes, noise spikes, and probabilistic output-spike decisions are
matched between methods. Different candidates have different initializations and
random streams. The root parameter initializer is used with checkpoint loading
disabled. Parameters are never updated; Adam is not applied.

For the full stimulus use `--duration-ms 2980`. The last 0.1 ms of the configured
2980.1-ms acquisition is omitted because it is not a complete loss bin. Duration
must be a multiple of 10 ms. GPU is supported via `--device cuda`, but CPU may
be competitive for tiny batches and these many small operations. CUDA has not
been validated by the CPU pilot. `--project-root` permits launching elsewhere.
The existing data loader assumes `dt=0.1` ms, which this experiment enforces.

The original E-prop code is replayed separately per candidate to give each a
distinct noise stream (its original noise interface shares noise across batches).
Autograd runs the candidates together as a batch. These are matched paired runs,
not simultaneous independent training optimizers.

## What is recorded

Each 10-ms emission is the signed parameter gradient of that bin's loss. BPTT
retains the entire preceding state graph, including delays and synaptic dynamics.
It is NOT 10-ms truncated BPTT. A new `autograd.grad` call computes each bin's
contribution without accumulating into `.grad`. Their sum is checked against one
backward pass on the sum of all bin losses.

The loss is `(count - target_count - 0.5)**2`, summed over independent candidates.
Counts sum all ten trials and all timesteps in the bin. This matches the existing
loss-handler update `2*(count-target-0.5)`, **not** the unshifted SSE it reports.
No batch mean is taken, so a candidate's gradient is not divided by batch size.
CV and whole-sequence firing-rate losses are excluded from this PSTH experiment.

The twelve parameter columns are:

1. Strf_gain
2. Strf_alpha
3. output_ad
4. on_ron_gSYN
5. off_ron_gSYN
6. on_sonoff_gSYN
7. off_sonoff_gSYN
8. sonoff_ron_gSYN
9. abs_ref
10. rel_ref_a
11. rel_ref_b
12. rel_ref_c

**Important:** your E-prop PSTH update only trains columns 1-8. Its refractory
parameters are trained by the separate CV objective. Consequently its last four
PSTH-gradient columns are zero. Autograd computes all twelve PSTH derivatives,
but the comparison plots use the eight common PSTH parameters and the five
conductances. Calling a twelve-dimensional comparison an equal-objective,
equal-routing comparison would be misleading.

Outputs in a new timestamped `results/` directory:

- `comparison.mat` and `comparison.npz`: bin and cumulative gradients, gradient
  norms, per-candidate cosine similarities, both rasters, parameters, and targets.
  Gradient array axes are `[emission, batch, parameter]`; raster axes are
  `[batch, trial, cell, timestep]`. No squeezing is required to retain batch IDs.
- `frozen_inputs.npz`: parameters, spikes, rates/partials, noise, and the actual
  uniform output draws used for the paired run.
- `gradient_alignment.png`: individual and cumulative cosine for each candidate.
- `rasters_batch_XXX.png`: matching output rasters for each candidate.
- `summary.json`: seeds, configuration, source hashes, versions, and validation.

Zero-norm comparisons are NaN, not perfect agreement or orthogonality. Cosine
distance is `1 - cosine_similarity`. Comparisons use raw parameter coordinates;
changing parameter units can change the eight-parameter cosine. The gSYN-only
comparison avoids mixing most of those units but does not solve every scaling issue.

## Surrogate assumptions and limits

The autograd forward pass uses hard spikes and the original ODE/event order.
SurrogateValue supplies a custom backward graph without rounding the hard forward
values. The root model supplies all static constants and initial states.

- Threshold: `q=(1+tanh((V-Vth)/5))/2`, derivative `sech^2((V-Vth)/5)/10`.
- Output spiking: hard threshold plus frozen Bernoulli draw, backward through
  `q * probability * nonrefractory_indicator`.
- Relative refractory probability: clipped forward probability, unclipped
  parameter partials, as in root `conditional_handler.py`.
- Absolute refractory duration: hard forward clamp; tanh gate derivative with
  `abs_ref/dt`. Recorded last-spike times are discrete/detached.
- STRF: existing onset/offset rate partials are attached as a local linearization
  at the fixed initialization. Frozen Poisson draws use the `dt/1000` rate
  surrogate, not a derivative of the sampled random comparison. The offset rate
  and its derivatives come from the SAME root preprocessing call for both methods.
- Default `--reset-gradient detach`: detach spike decisions in voltage resets
  and adaptation increments. Voltage and adaptation state history itself remains
  connected. This matches the reset-stop approximation used for the main root
  sensitivity paths. `--reset-gradient full` differentiates those events too;
  it changes gradients but must not change any forward spike.
- Synapse event updates, release memory, facilitation/depression, and delay history
  use autograd chain rules through the surrogate spikes.

The hand-written root sensitivities are not one globally consistent Jacobian:
their reset/adaptation treatment differs by parameter, and their event-insertion
partials are not identical to differentiating the entire hard event update.
This script therefore matches the stated local surrogate primitives, not every
hand-written sensitivity recursion. It is a genuine reverse-mode autograd
reference under the rules above, NOT proof that the root forward-sensitivity
code computes identical derivatives. The July E-prop post-reset eligibility and
hidden-layer feedback factor are kept unchanged, including their approximations.

The script refuses to draw alignment plots if the paired forward rasters differ
or the maximum voltage discrepancy exceeds 0.002 mV. It saves diagnostics first.
Silent members are flagged. It checks finite gradients and loss decomposition.
Running backward once per bin retains a large graph and revisits prefixes;
runtime grows roughly quadratically with sequence length. Start with the pilot.

## Tests

```bash
python Archive/autograd_eprop_comparison/test_comparison.py
```

Tests cover hard-forward/smooth-backward semantics, cosine edge cases, isolated
batch gradients versus standalone candidates, reset-invariant forward spikes,
cross-bin state connectivity, and equality of summed bin/total gradients.

PyTorch API reference:
https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
