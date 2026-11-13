# AJILE-12 pre-training data

Use the existing pre-training experiment with `+data=ajile12`. Dataset source,
audit directory, and pinned test path are local settings. Storage continues to
use the shared machine configuration.

The adapter preserves the released 60 Hz harmonic notch filtering and
within-group median reference, documented in the
[AJILE12 descriptor](https://www.nature.com/articles/s41597-022-01280-y).
It does not apply another filter or reference. Every model example is one
electrode, resampled from 500 to 2048 Hz with rational polyphase filtering.
Five-second native windows map to 10240 samples before the existing
BrainBERT one-sample trim, STFT and masking. Valid neighboring windows supply
resampling context; excluded spans are never used as context.

Run the full audit with `python -m data.ajile.audit --config CONFIG`.
It scans every sample, stores resumable reason masks in per-session blocks,
and excludes bad electrodes, any blocklist overlap, nonfinite and exactly
constant electrode windows. Incomplete trailing samples are counted and
discarded. Invalid electrode-table mappings or missing groups fail explicitly.
No additional group-size exclusion is applied to already-referenced signals.

Per-session summaries and the final inventory report eligible examples,
recording seconds with at least one eligible electrode, electrode-weighted
seconds, electrode/group/activity counts, and independent plus exclusive
rejection counts. Activity memberships may overlap and must not be summed as
exclusive categories. Combined activity labels are reported separately.

Whole sessions are assigned by minimizing the sum of absolute deviations from
98/1/1 eligible-example fractions, retaining a training session for every
subject. Sorted input and a secondary session-order objective resolve ties.
Only test sessions and their source-content identities are pinned. The loader
requires the completed inventory; it cannot train against a partially scanned
corpus. Source size and modification time guard reuse of a completed scan;
content identity includes metadata and hashes of all decoded source blocks.
No spectrogram corpus is generated during scanning. Existing feature caching
is used lazily by training.

AJILE STFT normalization uses one mean and population standard deviation per
frequency and electrode, fitted across eligible windows in that recording's
assigned split. Training recordings use only training windows. Validation and
test recordings use their own unlabeled split windows; current splits assign
whole sessions, so no session contributes to more than one split. Excluded
windows never contribute. The five-second waveform trim and STFT edge trim
match the features supplied to the model.

The recording-statistics cache lives under the configured feature cache root,
with a resumable per-electrode partial and one complete table per session.
It stores only moments and frame counts, not spectrograms. Preparation can be
run separately with `python -m data.ajile.normalization --source-root ...
--audit ... --feature-cache ... --preprocessor ...`; pass the local dataset,
audit and cache roots and the STFT YAML. Training also prepares a missing
recording cache lazily, which may delay its first batch. The full AJILE pass
covers tens of millions of electrode windows and should be run detached under
the normal CPU priority and memory limits before a large training campaign.

The optional `dataset_zscore` mode pools eligible training windows across
recordings with the same subject and electrode ID. It fits one mean and
standard deviation per frequency for each electrode. Validation and test
recordings use these training moments, and contribute no frames. A held-out
electrode without training statistics raises an error. The pooled cache is
keyed by the complete training-recording list and STFT settings. The fitted
arrays and electrode identities are also persistent buffers in the model
checkpoint; `normalize_stft` can apply a checkpoint's moments to an unmasked
STFT at inference. Prepare this mode with the normalization command above
and `--normalizing dataset_zscore`.
