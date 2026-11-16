# Scientific training monitors

BrainBERT predicts the magnitude of selected spectrogram cells. All loss
values below refer to the preprocessed spectrogram, so their units follow the
configured STFT normalization rather than the raw signal voltage. Training and
validation diagnostics are recorded at the same completed-update steps.
Training summaries cover batches since the preceding evaluation step;
validation summaries cover the current evaluation pass. Epoch summaries cover
all training batches in one data pass and validation at its final checkpoint.
The test pass uses the checkpoint selected by validation loss.

With the default recording-wise STFT z-score, both phases use deviations
from an electrode and frequency mean as their scale. Each recording fits
its own moments from eligible windows in its assigned split, so one
normalized unit can represent a different raw magnitude across
recordings. The monitor metadata identifies the normalization mode.

## Epoch summaries and representation diversity

`epoch_metrics.csv`, `epoch_mask_groups.csv`, and
`epoch_frequency_coverage.csv` summarize the same diagnostics across a
complete data pass (or a marked partial final pass). Training values are
online averages from weights changing during that pass. Validation values
use the single checkpoint at its end; their difference is therefore not a
same-checkpoint generalization gap.

`validation_rank.csv` uses a fixed, evenly spaced subset of validation
windows at every evaluation. It averages non-padding frames of the final
transformer layer for each window, centers those vectors, and reports:

| Diagnostic | Calculation | Scientific meaning |
| --- | --- | --- |
| `effective_rank` | Exponential entropy of normalized covariance eigenvalues | Number of comparably active embedding directions. |
| `effective_rank_fraction` | Effective rank divided by `min(windows - 1, hidden size)` | Diversity relative to the attainable rank. |
| `pca95_components` | Smallest number of principal components explaining at least 95% of variance | A complementary threshold-based dimension count. |

These embeddings come from the same masked inputs used for validation
reconstruction. A zero-variance embedding set has rank zero.

## Per-frequency validation reconstruction

`validation_frequency.csv.gz` reports each STFT frequency separately for
all selected validation cells and for temporal-only, spectral-only, and
overlap selections. Each row provides selected and content-cell counts,
mean L1, content L1, and L1 relative to the zero-prediction baseline.
All values use the existing validation masks and forward pass. Their
frequency-dependent masking difficulty remains visible through the
selection groups and counts; they are not isolated single-frequency probes.

## Training objective

Let `target` be the unmasked spectrogram value, `prediction` the model output,
and `alpha` the configured content weight. Only selected cells contribute.

| Diagnostic in `metrics.csv` | Calculation | Scientific meaning |
| --- | --- | --- |
| `original_l1_loss` | Mean `abs(target - prediction)` on selected cells within each batch, then mean across batches | Ordinary reconstruction error in the objective. |
| `original_content_aware_loss` | Within each batch, mean absolute error on selected cells with `abs(target) > 1`, multiplied by `alpha`; then mean across batches | Extra weight given to high-magnitude targets. |
| `original_loss` | Sum of the two original batch losses, averaged across batches | The optimization objective; validation values select the best checkpoint. |

The pooled `l1` and `weighted_content_l1` columns in `metrics.csv`
are calculated across all selected cells in the interval or evaluation pass.
They can differ from the original batch-mean objective when batches contain
different numbers of selected cells.

## Mask-conditioned reconstruction

`mask_groups.csv` reports the nine combinations of **selection strategy** and
**replacement action**:

| Selection strategy | Meaning |
| --- | --- |
| `temporal_only` | Selected by temporal masking alone. |
| `spectral_only` | Selected by spectral masking alone. |
| `overlap` | Selected by both; counted once in reconstruction loss. |

| Replacement action | Meaning |
| --- | --- |
| `zero` | Last replacement set the input cell to zero. |
| `random` | Last replacement copied another value; that source may itself have been masked. |
| `unchanged` | Neither strategy replaced the selected cell. |

A later masking operation can overwrite an earlier one. A later decision to
leave a cell unchanged does not undo an earlier replacement. Rows with `all`
in one dimension pool the corresponding groups across that dimension.

For any group, let `N` be its `count`, `E` its `absolute_error_sum`, `Nc` its
`content_count`, and `Ec` its `content_error_sum`. The zero baseline always
predicts zero, the fitted mean of a frequency within an electrode under the
recording and training-dataset z-score modes:

| Diagnostic | Calculation | Scientific meaning |
| --- | --- | --- |
| `l1` | `E / N` | Mean error for cells exposed to this masking condition; its scale does not increase merely because more cells were selected. |
| `content_l1` | `Ec / Nc`, using only targets with `abs(target) > 1` | Reconstruction error on the group's high-magnitude targets. |
| `weighted_content_l1` | `alpha * content_l1` | That group's content-aware term at the configured weight. |
| `zero_baseline_l1` | Mean `abs(target)` over the same `N` cells | L1 error of the zero prediction on this selection. |
| `relative_l1` | `l1 / zero_baseline_l1` | Reconstruction error relative to the constant prediction: below 1 means the model reconstructs better than zero. |
| `content_zero_baseline_l1` | Mean `abs(target)` over the same `Nc` high-magnitude cells | Baseline error on exactly the content subset. |
| `relative_content_l1` | `content_l1 / content_zero_baseline_l1` | High-magnitude reconstruction error relative to the same baseline on that subset. |
| `target_mean`, `target_std` | Mean and population standard deviation of the group's targets | The target level and variability under this masking condition. |
| `prediction_mean`, `prediction_std` | Mean and population standard deviation of the group's predictions | Prediction bias and variability; a small prediction standard deviation relative to targets may indicate collapse toward a constant. |

`metrics.csv` reports the same pooled diagnostics over **all** selected cells.
Its `original_relative_l1` and `original_relative_content_l1` also compare
the original batch-averaged L1 components to batch-averaged zero-baseline
errors. Multiplying numerator and denominator by `alpha` would leave the
content ratio unchanged.
Group and overall means are formed from error sums and their own cell counts,
not from an unweighted mean of batch or group means. Counts show how much
evidence supports each conditional estimate. Differences between groups may
reflect both model behavior and the difficulty of their masking conditions;
these rows are not isolated temporal-only or spectral-only experiments.

## Optimization dynamics

These `metrics.csv` measures accompany the reconstruction losses:

| Diagnostic | Calculation | Scientific meaning |
| --- | --- | --- |
| `lr_group_*_last_used` | Learning rate applied to the last update in the training interval, for each optimizer parameter group | Relates changes in loss to the learning-rate schedule. |
| `gradient_norm_mean` | Mean pre-clipping gradient norm over updates in the interval | Typical gradient scale during optimization. |
| `gradient_norm_max` | Largest pre-clipping gradient norm in the interval | Detects occasional large gradients hidden by the mean. |
| `clipping_fraction` | Fraction of interval updates whose pre-clipping norm exceeded the configured clipping threshold | How often gradient clipping constrained updates. |

## Frequency masking coverage

`frequency_coverage.csv` uses the actual STFT bin centers in Hz. It measures
**how often** each frequency was selected, not its reconstruction error.

| Diagnostic | Calculation | Scientific meaning |
| --- | --- | --- |
| `selected_cells` | Number of selected time-frequency cells at the bin | Absolute exposure to the specified masking strategy. |
| `valid_cells` | Number of non-padding cells available at the bin | Exposure denominator. |
| `selection_rate` | `selected_cells / valid_cells` | Fraction of available cells selected; compare across frequencies to assess masking balance. |

The `temporal_only`, `spectral_only`, and `overlap` series partition selected
cells. `temporal` and `spectral` each include overlaps; `union` counts each
selected cell once. Training has both interval and cumulative coverage, while
validation and test coverage describe their evaluation passes. A count can
rise simply because more windows were observed, so use `selection_rate` when
comparing frequencies or passes.

TensorBoard labels the actual batch-mean optimization objective as
`train/overall/total_loss` and `validation/overall/total_loss`.
It shows one overall series for each other diagnostic and six margin
summaries: three selection-strategy totals and three replacement-action
totals. Each margin shows count, content count, L1, content L1, and their
zero-baseline-relative versions. The full nine-cell grid, raw aggregation
sums, and per-frequency rows remain in the CSV tables. Heatmaps and frequency
coverage figures remain available. A configured `train_monitor_step` writes
training-only summaries between validation passes without evaluating the
validation set; the AJILE pilot uses 100 optimizer updates.
