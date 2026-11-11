# Experiment artifacts and logging

Keep the existing experiment configuration (`+exp=spec2vec` for pre-training)
and scientific groups. Add `+local=machine.local` to use the ignored shared
`conf/local/machine.local.yaml` overlay. It contains only machine storage,
logging, and runtime settings, shared by pre-training and fine-tuning.
Campaign name, protocol revision, data manifest and pinned test file are
separate experiment inputs. Supply them explicitly with `+campaign=NAME`,
`+protocol.revision=REVISION` and `+data.pinned_test=FILE` or a campaign overlay.
Set the CPU thread limit and GPU allocator limit after profiling the intended
model and batch. No optimizer or batching settings live in machine config.

The shared storage and runner modules are task-independent. Managed fit reuse
currently validates file-manifest datasets; other datasets need a sample
identity adapter as part of the separately planned data-loading work.

The optimizer, batching, masking, random-state behavior, preprocessing numerics,
and validation schedule remain those of the existing pipeline.

Reusable checkpoints live in `fits/<scientific-hash>/checkpoints` beneath the
artifact root. Campaign directories contain references, not copies. Completed
fits are checked before reuse; incomplete attempts are quarantined on retry.
Checkpoint replacement stages, backs up, audits, and atomically installs the
new file before removing the backup. Only best and last checkpoints are kept.

Scientific identity includes model, objective, preprocessing, optimizer,
budget, batch sizes, explicit protocol revision, source content checksums, and
actual split membership. Device count, workers, storage paths, and output
verbosity are excluded. Existing `log_step` and `checkpoint_step` are included
because the original runner uses both to trigger validation and best-model
selection. Changing this coupling would require a separate scientific change.
Scientific implementation changes require an explicit protocol revision,
not a repository-wide source hash.

The pinned test JSON is a nonempty list of manifest-relative sample names;
current generated test membership must match exactly. Training and validation
membership is recorded but need not be separately pinned. No new production
split or data selection is created automatically.

STFT caching stores finite unmasked tensors beneath the configured cache root,
keyed by waveform content, preprocessing settings and cache revision. Existing
STFT numerical behavior is preserved. Corrupt entries fail explicitly.

Invocation logs use `<log-root>/<campaign>/<invocation>/status`, `training`,
and `errors`; checkpoints and TensorBoard data belong under the artifact root.
Normal runs disable progress bars and TensorBoard images. Failed runs retain a
compact error record. CPU work runs at reduced priority. GPU memory fractions
limit the PyTorch allocator, not total process or host memory.


Audit status files report temporal progress in seconds, not native sample
counts. Group corpus totals, per-subject counts, current-recording duration,
and training-example counts separately. Each example count specifies its
unit and window duration. Remaining counts include active work. Apply this
convention to future audits as well.
