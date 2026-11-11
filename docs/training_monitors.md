# Scientific training monitors

Pre-training records diagnostics from its existing batches and evaluation
passes. Enable TensorBoard using `runtime.output_tb: true` in the shared local
machine configuration. CSV is always written. These settings do not change
fit identity or cause completed fits to be retrained.

## Artifact tables

Each fit has a `monitoring` directory beside `checkpoints`:

| File | Contents |
| --- | --- |
| `metrics.csv` | Original losses, pooled diagnostics, zero-prediction baseline, target/prediction moments, learning rates, gradient norms and clipping fraction |
| `mask_groups.csv` | Nine strategy/action groups, plus three strategy and three action marginals; error sums, counts, normalized losses and moments |
| `frequency_coverage.csv` | Frequency in Hz, selected-cell counts, valid-cell exposures and selection rates; interval/pass and cumulative training coverage |
| `metadata.json` | Metric definitions, frequency axis, content threshold and weighting, category semantics |
| `tensorboard/` | Matching scalar curves, conditional-L1 heatmaps and frequency coverage bars |

Rows identify phase, completed optimizer update, interval boundaries and the
evaluated checkpoint step. Training records pool all batches since the last
logging boundary; these predictions were produced by evolving weights.
Validation pools each existing validation pass. Final test diagnostics use
the validation-best checkpoint. Coverage counts observations, not unique
examples across repeated visits. No extra evaluation pass is introduced.

## Reading conditional losses

Selection categories are temporal-only, spectral-only and overlapping masks.
Replacement actions are zero-fill, random replacement and no replacement.
The action is the last actual write: a later no-op preserves an earlier
replacement, and a randomly copied source may already contain corruption.
A coincidental numerical match with the target does not change the action.

Every group's L1 is its absolute-error sum divided by **its own selected-cell
count**. Its content L1 uses its own subset with `abs(target) > 1`; the weighted
term multiplies that mean by the configured alpha. The diagnostic objective
adds these two conditional terms. Missing denominators produce empty CSV
fields and no TensorBoard scalar, never zero loss. Marginals pool sums/counts,
not group means. Padding is excluded, and overlapping cells count once.

The original objective and checkpoint-selection metric retain their original
batch-mean reductions. The pooled diagnostic objective can differ and must
not be substituted for checkpoint selection. Group objectives are not
additive contributions to the original objective.

Normalization removes scaling by the number of masked cells. Sample and mask
composition can still vary between steps. Counts and target moments help
interpret this variation; the groups do not represent isolated masking
experiments. With z-scored STFT input, errors describe standardized magnitude,
not voltage. The zero-prediction L1 is a reference on the same selected cells,
not an additional training objective.

Frequency bars show masking exposure, **not reconstruction loss**. Their axis
uses actual STFT bins, rather than the grid used internally to choose mask
widths. Variable masking and boundary handling need not yield uniform rates;
monitoring does not rebalance them.

CSV files and TensorBoard events are flushed at existing logging boundaries
and before completion checksums. Completed fits are immutable; older fits
without monitors remain reusable. Failed attempts follow shared quarantine
rules. No device throughput, resource measurements, parameter histograms or
per-frequency reconstruction losses are added by these monitors.
