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
