# Pretraining reports and test diagnostics

Reusable profiles live in `conf/reports/pretraining.yaml` and
`conf/diagnostics/pretraining.yaml`. AJILE opts into them through Hydra defaults;
other experiments can reference the same profiles. Generic experiment defaults
do not enable reports or diagnostics.

Reports explicitly select formats (`png`, `csv`), splits, and figure/table
families. Test diagnostics explicitly select `representation_rank` and/or
`frequency_errors`. Missing, null, or empty selections produce no corresponding
outputs. Booleans are rejected. These operational selections do not change fit
identity.

Reports are saved under the campaign's `reports` directory. PNGs are grouped by
`training`, `validation`, and `test`, with one `mask_conditions` subfolder per
split. Its `margins_` files show six row/column totals; `full_grid_` files show
all nine conditions. Other figures share the split folder. CSVs retain their
split and diagnostic-family organization. `output_index.csv` records output
paths and availability. Ready saved-data products are published before test
diagnostics finish.

To render saved results, run `python -m util.experiments.reporting` with
`--artifact-root`, `--campaign`, and `--report-config` pointing to the report
profile. Existing fits without saved selections require an explicit profile.
For a completed STFT fit missing test diagnostics, use
`python -m util.experiments.test_diagnostics` with the same artifact/campaign
arguments plus `--diagnostics-config` and `--report-config`. The repair reuses
the frozen test selection and selected checkpoint, verifies saved test metrics,
and writes derived outputs without changing the completed fit. Verified cached
diagnostics are reused.

Validation and test share the frequency accumulator and representation-rank
calculation. The rank calculation samples up to 2,048 evenly spaced windows.
`total_loss` retains the training objective's batch-mean definition; other error
ratios use pooled cell totals. Histories use actual optimizer steps and discard
superseded unsaved intervals after checkpoint restoration. Figures describe one
fit and do not establish statistically significant model differences.

Single-checkpoint test rank and overall relative errors are exported as CSV
tables only. Validation rank remains a figure showing its training trajectory.
Figure legends are shared across panels with matching series. E1, E2, etc.
mark completed epochs; the dotted vertical line marks the selected checkpoint.
The constant prediction L1 baseline predicts zero at every standardized cell.

Frequency reconstruction figures contain raw L1 and content L1 only. Separate
`frequency_reconstruction_normalized.png` figures divide each loss by its
matching constant-prediction loss, with a reference at one. Both figures identify
the validation-selected checkpoint in their titles.

Future STFT experiments use the reusable `conf/evaluation/pretraining.yaml`
profile for two separate test evaluations: 100% zero replacement and 100%
random replacement. Training and validation retain their original masking.
Checkpoint selection therefore remains based on the original validation loss.
Replacement actions describe writes, not guaranteed value changes: a random
replacement can coincidentally equal the original value.

To reevaluate an existing checkpoint, add `--evaluation-config` and
`--evaluation-name zero_only` or `random_only` to the test-diagnostics
command. Each evaluation retains the frozen test-window membership and writes
its own results, monitor tables, policy, checkpoint digest and completion
checksums under the campaign's `evaluations/<name>` directory. Derived reports
use the shared `reports/figures/test` directory, with policy suffixes on
frequency figures. Zero and random conditions share one heatmap under
`reports/figures/test/mask_conditions`. Tables remain separate by policy
under `reports/tables/test/<name>`, with a combined mask-condition table. The original fit and test results remain
immutable. Repeat invocations verify and reuse completed evaluations.


### Fixed-mask test diagnostics

The reusable `conf/diagnostics/fixed_masks.yaml` profile is selected with
`--diagnostics-config` when replaying a completed fit. It leaves training,
validation, and checkpoint selection unchanged. Missing or empty `fixed_masks`
disables this diagnostic. Every temporal, spectral, and joint condition uses
the complete pinned test benchmark. Spectral-bin assignments are balanced and
reused across conditions; temporal gaps are nested at paired positions.

The duration comparison reports L1, content L1, and total loss relative to
the previous zero constant predictor, with raw objectives retained in CSV.
Total loss is normalized by the constant predictor's combined objective.
Frequency figures normalize errors by a frozen training-target mean for each
frequency; joint curves score the time–frequency intersection. Small condition
summaries and the training mean are cached under campaign diagnostics for
resumption. Figures share `reports/figures/test/fixed_masks/frequency`;
source tables reside under `reports/tables/test/fixed_masks`.
