# Experiment artifacts and logging

Keep the existing experiment configuration (`+exp=spec2vec` for pre-training)
and scientific groups. Add `+machine=machine.local` to use the ignored shared
`conf/machine/machine.local.yaml` overlay. It contains only machine storage,
logging, and runtime settings, shared by pre-training and fine-tuning.
Campaign name, protocol revision, data manifest and pinned test file are
separate experiment inputs. Supply them explicitly with `+campaign=NAME`,
`+protocol.revision=REVISION` and `+data.pinned_test=FILE` or a campaign overlay.
Set the CPU thread limit and GPU allocator limit after profiling the intended
model and batch. No optimizer or batching settings live in machine config.

## Multiple GPUs

Set `CUDA_VISIBLE_DEVICES` to the permitted devices and
`exp.runner.multi_gpu=true` to use the existing `DataParallel` runner.
Training and validation batch sizes are global: each batch is divided among
the visible GPUs, and the objective is evaluated on the gathered outputs.
Keep these batch sizes unchanged when resuming on a different device count.
Changing devices preserves fit identity and restores the saved optimizer
step, but does not promise identical dropout or worker random streams.

For hosts where NCCL device or shared-memory transport fails, set
`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`, and
`NCCL_CUMEM_HOST_ENABLE=0` in the launch environment. This selects the socket
fallback and must be configured before CUDA initialization. Machine-local
Hydra overlays can supply these variables through `hydra.job.env_set`.
Profile this fallback on the intended host; additional GPUs only accelerate
the model work, so data loading can still limit total training throughput.

## Shared artifacts

The shared storage and runner modules are task-independent. Managed fit reuse
currently validates file-manifest datasets; other datasets need a sample
identity adapter as part of the separately planned data-loading work.

The optimizer, batching, masking, random-state behavior, preprocessing numerics,
and validation schedule remain those of the existing pipeline.

Reusable checkpoints live in `fits/<scientific-hash>/checkpoints` beneath the
artifact root. Campaign directories contain references, not copies. Completed
fits are checked before reuse. An incomplete fit with matching last and
best checkpoints and an unshuffled training order resumes its model,
optimizer, scheduler, and step count;
invalid or uncheckpointed attempts are quarantined. Only the latest and best
checkpoints are retained, so an interruption can lose updates since the most
recent checkpoint. CSV monitoring preserves separate invocation segments.
TensorBoard uses one shared event directory per fit across all resumes,
connecting the saved history and hiding unsaved updates after rollback.
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
membership is recorded but need not be separately pinned. The existing step-based workflow creates no new production split or data
selection. AJILE budgeted selection is documented in
[AJILE-12 pre-training](ajile12.md).

STFT caching stores finite unmasked tensors beneath the configured cache root,
keyed by waveform content, preprocessing settings and cache revision. Existing
STFT preprocessing defaults to per-electrode, per-frequency recording-wise
z-scoring. Explicit `normalizing: window_zscore` retains the older window
behavior. Source recordings need explicit electrode and recording identities;
missing identities fail clearly. The normalization mode is a scientific fit
parameter and changes the fit hash. Corrupt cache entries fail explicitly.

Invocation logs use `<log-root>/<campaign>/<invocation>/status`, `training`,
and `errors`; checkpoints and TensorBoard data belong under the artifact root.
Normal runs disable progress bars. Scientific pre-training monitors write
compact CSV tables and TensorBoard curves under each fit's `monitoring`
directory; see [training monitors](training_monitors.md). Failed runs retain a
compact error record. CPU work runs at reduced priority. GPU memory fractions
limit the PyTorch allocator, not total process or host memory.


Audit status files report temporal progress in seconds, not native sample
counts. Group corpus totals, per-subject counts, current-recording duration,
and training-example counts separately. Each example count specifies its
unit and window duration. Remaining counts include active work. Apply this
convention to future audits as well.

Set `preprocessor.normalizing: dataset_zscore` to pool unmasked STFT frames
across all training recordings for each subject/electrode/frequency. The
fitted moments are stored as model buffers and used for training, validation
and test. Validation/test windows never contribute to fitting. Electrode IDs
must be present in training; missing IDs fail explicitly. The standard STFT
configuration continues to default to recording-wise normalization.


### Objective follow-ups

`ajile12_followup` uses a 20-epoch ceiling and four-epoch early stopping on
the experiment's own regular-validation objective. `content_reference` logs
both original and spectral content losses while optimising the original
objective; `spectral_content` optimises the normalised spectral weighted
mean. Both retain the outer content coefficient. Spectral weights use actual
STFT frequencies and configurable strength/exponent.

A continuation supplies a local `parent_checkpoint` and its scientific
`parent_checkpoint_sha256`; data membership, architecture and the original
objective are checked before restoration. Weighted experiments initialise
from scratch. Each new fit owns its events, latest/best recovery state and
weight-only completed-epoch snapshots. Original completed fits are immutable.

Regular validation runs at completed-step intervals. The reusable fixed-mask
profile adds epoch-wise validation diagnostics, with resumable condition sums
and a frozen training-mean baseline. These shared-reference diagnostics do not
select checkpoints or affect early stopping. Pending diagnostics and patience
are restored with the latest checkpoint.


## Downstream checkpoint evaluation

Use `python -m util.experiments.downstream_evaluation` with the
brain_decoding task YAML, `--checkout`, `--output-root`, and
`--manifest-root`. Run it in the brain_decoding Python environment.
This BrainBERT-specific entry point uses `conf/evaluation/downstream.yaml`:
400 scored windows per split for RMS, finger regression, and Stanford finger
flexion; all other tasks retain their full tests. `--full-test` explicitly
selects full evaluation. Ordinary brain_decoding entry points remain full-set.

Share one manifest root across checkpoints. Temporal blocks and their smoothing
context are deterministic. Inference processes only selected windows and needed
context, reusing configured raw STFT caches. Existing readouts are restored when
the source configuration points to completed component fits; a separate output
root preserves full-test results. Paired comparisons must use the same manifests.
