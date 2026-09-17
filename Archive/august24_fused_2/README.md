# August 24 fused-loss revision 2

This is an isolated copy of `august24_local_eligibility_fused_loss`; that folder is unchanged.

The PSTH surrogate objective, rate objective, and CV objective are now mathematically paired with their gradients. CV uses an 11-step soft spike-time window and online pooled within-trial ISI statistics, so its memory use is fixed rather than proportional to the spike count. Refractory gates are shared by the forward spike draw and all eligibility paths, eligibility is evaluated before reset, hidden `Bk` is applied once, and adaptation uses a decaying event trace.

`surrogate_width_mv` controls the normalized voltage surrogate. Relative-refractory probability is `0.5*c*(1+tanh(a*delta-b))`, making `c` the maximum probability; this intentionally differs from the old expression whose maximum was `2*c`. Upstream CV derivatives remain local E-prop surrogates, not exact full-network forward sensitivities or saltation derivatives.

Run the focused checks with `python -m unittest discover -s tests -v`.
