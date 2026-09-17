# Nonlinear extension: outcome

Seven nonlinear classifier families were tested on all 12 fitted parameters, without a two-dimensional bottleneck: Gaussian SVM, asinh-transformed Gaussian SVM, polynomial SVM, k-nearest neighbors, random forest, boosted trees, and a small neural network. There were 38 predefined configurations. The 196 known-layer cells and SPIKE-selected batches match the earlier analysis; the 24 unknown-layer cells were excluded from training/evaluation.

## Main result

Boosted trees were the strongest family numerically, but did not establish convincing layer separation:

- Repeated cell-level nested CV: **35.0% balanced accuracy**, versus 32.5% for the earlier PCA baseline and 33.3% chance balanced accuracy.
- Boosted-tree results across the five cell splits: **38.1%, 29.7%, 32.7%, 33.9%, 40.3%**.
- The initial subject-grouped result was 41.9%. A follow-up check across five subject-grouped assignments gave **41.9%, 31.5%, 32.8%, 37.6%, 36.9%**, averaging **36.2%** (split SD 4.1 percentage points).
- Other nonlinear families averaged 29.1–31.6% balanced accuracy on the repeated cell folds.
- A procedure that selected both the family and its tuning parameters entirely within training folds scored **32.0%** balanced accuracy. Thus choosing the best family after seeing the outer results should not be mistaken for a validated improvement of the whole search procedure.

Balanced accuracy is the mean of the three per-layer recalls. Ordinary accuracy differs: boosted trees averaged 32.2%, and the highest ordinary accuracy among the tested families was 35.9% (asinh Gaussian SVM). Always predicting the most common layer would give 40.3% ordinary accuracy but only 33.3% balanced accuracy.

The repeated subject analysis is a post-exploration sensitivity check for the most promising family, not a new independent test or a selection-adjusted significance test. Repeated splits reuse the same cells; their SD is not an independent-sample confidence interval. No new permutation test was run. The small numerical gains need independent confirmation before being interpreted as reliable layer information.

## Reproduce

From this Analysis folder in MATLAB:

```matlab
results = explore_nonlinear_layer_classifiers;
validate_nonlinear_layer_analysis;
stability = check_boosted_subject_stability;
```

The main comparison uses five repeats of nested five-fold cell CV, with three inner folds and training-only preprocessing. It reuses the earlier outer cell partitions. Subject validation uses subject-balanced five-fold outer partitions and subject-grouped three-fold inner partitions; this differs from the earlier leave-one-subject-out analysis.

Validation passed for original cell split agreement, disjoint training/test rows, subject isolation at both nesting levels, training-only family choice, recomputed balanced accuracies, and reusable model predictions. The first subject-stability repeat exactly reproduced the original grouped boosted-tree predictions.

Outputs are in `nonlinear_layer_results`: `README_results.md`, `method_comparison.csv`, `nonlinear_comparison.png`/`.fig`, `nonlinear_analysis.mat`, `boosted_subject_stability.csv`/`.mat`, and `validation_log.txt`. The main comparison plot shows the **initial single subject-grouped assignment** for all families; the additional boosted-tree repetitions are in the stability files and summarized above.

The saved `results.finalModel` is selected by CV on all known cells, which selected nearest neighbors for the final descriptive fit. It is not necessarily the family with the highest outer-CV average. To predict new parameter rows in the original units/order (`results.dataset.names`):

```matlab
[indices,scores] = predict_layer_classifier(results.finalModel,Xnew);
predictedLayers = results.classes(indices);
```

Existing source data and earlier analysis scripts were not changed.
