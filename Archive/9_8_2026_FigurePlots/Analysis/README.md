# Parameter-to-layer exploration

Run in MATLAB from this folder:

```matlab
results = explore_layer_parameter_space;
validate_layer_analysis;
```

The source is `100_epoch_all_cells_Eprop.mat`, using all 12 final-epoch parameters and one batch per cell selected by the SPIKE-distance rule in `ParameterMDS.m`. Existing analysis scripts and source data are not modified. A timestamp/size-checked cache avoids repeating SPIKE selection. If the selection code changes, remove only `layer_parameter_results/selected_parameters.mat` to rebuild the cache.

## Result of the September 14, 2026 run

None of the five tested two-dimensional methods reliably separates the three known layers on held-out cells. Mean balanced accuracy was 32.5% for PCA, 29.6% for LDA, 29.6% for shrinkage LDA, 29.2% for asinh-transformed shrinkage LDA, and 30.2% for a nonlinear RBF Fisher projection. Chance balanced accuracy is 33.3%. Balanced accuracy is the average of the three within-layer correct-classification rates; it weights each layer equally.

Leave-one-subject-out results were also near chance (27.6–31.2%). Subject groups are taken literally from the dataset's `subject` field; their interpretation as distinct animals depends on the metadata. A 99-shuffle, five-method maximum-statistic permutation check yielded p = 0.81. This is a cell-exchangeability check, not animal-level inference.

This is evidence against useful separation by these methods for this run, not proof that no layer information exists in any fitted model or nonlinear relationship. The unknown-layer cells were excluded from fitting and evaluation, then displayed as gray crosses. Full-data maps and their decision regions are descriptive training fits, not held-out maps.

Independent validation passed: saved cell IDs and layers remain aligned with source rows; parameter values match the selected source batches; the reusable projection function reproduces saved coordinates; and custom LDA predictions matched MATLAB `fitcdiscr` with uniform priors exactly in every tested fold across five repetitions.

## Files

- `explore_layer_parameter_space.m`: comparison, nested validation, plots, permutation test.
- `prepare_layer_parameters.m`: source extraction and SPIKE batch selection.
- `project_layer_parameters.m`: transform new parameter rows using a saved projection.
- `validate_layer_analysis.m`: independent alignment and built-in LDA checks.
- `layer_parameter_results/README_results.md`: full numerical results and caveats.
- `layer_parameter_results/01_projection_comparison.fig`: editable MATLAB maps and decision regions; also PNG/PDF.
- `layer_parameter_results/02_heldout_confusions.fig`: actual held-out confusion matrices; also PNG.
- `layer_parameter_results/03_coefficients_and_controls.png`: projection coefficients and sensitivity controls. Coefficients are descriptive, not evidence of significant parameter effects.
- `layer_parameter_results/layer_analysis.mat`: results structure, including models, scores, out-of-fold predictions, and permutation results.

Retrieve cell coordinates (original cell order) with `results.scores{2}` for LDA. Project new rows using `project_layer_parameters(results.models{2}, Xnew)`; columns must follow `results.dataset.names`, in original parameter units.

The initialization control is the first saved epoch of the final-selected batch, not necessarily the state before training. The median-batch control uses the final parameter-wise median across all 12 batches. Neither control improved the substantive conclusion.
