# Agents.md

Please use the conda environment specified in local configurations (e.g., `.vscode/settings.json`) for testing codes.

## FIRST PRINCIPLE -- GRILL ME

You should always grill me on ALL technical details and design choices rather than making guessworks or silently setting default values by yourself.

Anti-examples: 

- I asked you to design the figures/plots, and you silently designed the content of one or several figures without checking details like x-y axis meanings, axis ticks, legends, plot timing with me.
- Silently save logs / artifacts of a new experiments with those of previous experiments without asking whether the save path should be changed.

## Coding style

### Writing Style

- All lines with longer than 80 characters should be broken into several lines. (This limit excludes markdown files, html, and comments) 
- Clear unused indents. 
- When printing the path to a file, NEVER use relative path.

### No Repeat

- Remove duplicate or extremely similar code blocks. Create new functions or classes if necessary! 
- Share common logic through abstraction.
- Remove unused imports, functions and variables.

### Handle Edge case

- Warnings and errors should be raised in edge cases (e.g., empty rows), don’t do the following: produce an empty output file, generate empty column, set a value to default (e.g., 0 / 1),  make assumptions by yourself.  
- All errors and warnings messages should be clear. For example, instead of saying “shape error”, say “expected shape {}, but got {}.” 
- NaN should be checked before returning the values.

### Constants over magic numbers

- Replace hard-coded values with named constants at the top of the file. 
- Keep constants at the top of the file, or in cases constants are shared across multiple files, save in a dedicated constants file.

### Comments and Docstrings

- Add proper numpy style docstrings and type annotations to functions. You should clearly specify the shape of arrays and tensors. Add ",optional" mark and default value for optional variables. 
- Document APIs, complex algorithms, and non-obvious side effects etc.
- Don't comment on what the code does (codes should be self-documenting)
- DONT paraphrase variable names, you should check how the variables are actually used in the workflow.
- Explain the usage, input, and output at the top of the file. Input and output includes file structure, contents of each file (table-formatted files or files with multiple keys should be explained in details). You do NOT need to provide example rows.

### Single Responsibility

- Each function should do exactly one thing. Bud don't create wrapper function for a function from another library. 
- Functions should be small and focused
- If a function needs a comment to explain what it does, it should be split.
- Functions that are general and reusable by other classes should NOT be class functions.

## Data

The train-test splits of a task should ALWAYS MATCH EXACTLY across models. This is a MANDATORY check before launching each model's full fitting compaign -- the outer split list match previous model fits EXACTLY. ANY model configuration should NOT change the definition of outer splits in ANY way -- that is, the test set MUST remain the same. 

- However, new splits can be added with explicit approval.
- Test rows MUST come from the same window (timestamp and label), but can differ in preprocessing and minor window length differences.

Train-test splitting procedure should be 100% deterministic. NEVER add randomness that varies across experiments in ANY step of sample preparation nor splitting.

## Smoke tests

I request a FULL smoke test overing ALL tasks to be tested by the compaign -- that is testing one train-test split for EACH task with MINIMAL training budget (e.g., 1 epoch, one randome seed, no HPO, one fold in CV), and check its artifacts.

- If smoke test reveals a problem, you should ONLY rerun the affected cases rather than rerunning the whole test suite after fixing.
- NEVER implement a computationally heavy smoke test that takes more than 10 minutes to finish.

Recording and split list check: apart from the one-split per task check above, please also check the ENTIRE configured recording list and the resulting outer split list for consistency with pinned split list, but WITHOUT fitting any model.

After ensuring enough coverage of the benchmark, you should try to reduce the computational cost of smoke tests.

Smoke test results and logs should be REMOVED immediately after validation.
DO NOT keep smoke test results until the end of official model fit.

## Experiment launch and hand-off

Whenever you are asked to launch a compaign of experiments, please detach and leave it running after smoke test and confirming one fit works successfully.

- DONT save artifacts on the one-fit audit, it is for you to check whether the training works.
- For multi-stage, large-scale pre-training, you DONT need to audit until the last stage. Auditing few updates of the first stage suffices.

If possible, report estimated time to finish after detaching.

## Experiment parallelisation

For the CPU tasks, lower-priority should be assigned to avoid the laucnhed python programme interfering with the server's core operational programmes.

You should parallelise independent neural fits whenever possible. Independent experiments should ALWAYS be launched without waiting other (or even a single stage of ) irrelevant experiments to finish.

- If there are multiple levels of parallelisms possible, select a level that doesn't create expensive cross-device communication overhead (e.g., if you assign batches across GPUs, this may have overhead)
- Ensure that you select the minimal level permitted by task independence that doesn't create communication overhead to maximise GPU usage and avoid a single bottleneck holding the entire compaign (e.g., multi-session experiment with weight sharing cannot be parallised at session-level, but session-independent model fitting experiments can)

For multi-stage experiment with cross-stage dependencies, DONT parallelise.

## Computational efficiency and memory usage

Cache and reuse intermediate results that are shared for multiple times. This includes cleaned neural recordings, features extracted by pre-trained mdoels used for several times, models sharing the same initial stages etc.

Always launch experiments (no matter GPU or CPU) with an appropriate memory cap measured by profiling the model, and launch with low-priority for CPU tasks. (AVOID oversubscription of the host's CPU or CUDA devices)

- DONT request a device with over-conservative GPU memory vacancy. For example, for a model with peak usage at 1GB, requesting a device with 16GB of free memory is not appropriate.

## Module disentanglement

REDUCE code repeat.

DONT create a completely new module / pipeline for every minor modification to the current experiment pipeline. Try to extend existing modules / functions to support an additional run mode if possible.

But DONT squeeze configurations of different compaigns together.

Creation of new files and modules should ONLY be permitted if all or most steps of the new pipeline is completely different from existing ones in the repository. If there are some steps to reuse from existing modules, please import rather than rewriting.

## Configurations and compaign identity

ONLY hash a model's fit using its scientific parameters (e.g., batch size, number of epochs, architectural parameters).

NEVER use invocation parameters (number of devices, GPU/CPU settings, etc.) that doesn't affect the scientific definition of model fits to generate hash identity.

NEVER use the entire python tree to hash the fit identity. If implementation of a scientific module changes and that has significant effect on the results (test by minimal validation), then reflect this scientific change in the configuration.

Changing the hashing mechanism or minor code implementations (e.g., file tree organisation) should NOT invalidate previous completed fits.

Configurations under `configs` are intended to save human readable experiment configurations, NOT machine-generated hashes.

## Break-point restoration

All experiments should store artifacts immediately after the results of one fit (the smallest unit of experiment, e.g., be one neural or non-neural fitting of a single random seed) are obtained to avoid losing results of already completed runs. However, failed, invalid, or interrupted run artifacts should be removed or quanrantined in the next invocation, starting a fresh experiment on that fit.

Artifact protection: if an old artifact is to be replaced by a new one (e.g., new fitting results, new checkpoint), the correct order is: backup old file -> save new file -> audit and then remove old file. NEVER remove old file before saving the new file.

If the failure in one experiment is not fatal (definitely affects the following fits), please report failure and continue trying following experiment rather than stopping the whole process.

NEVER retrain a model on a data / fold / session that already have completion records under the same configurations. (changines in invocation parameters like number of GPU/CPUs used should NOT restart already completed experiments)

### Pre-training / long-epoch training restoration

For the same training configurations (batch size, learning rate schedule, sampling strategy, data cohort, etc.), but larger number of epochs, please resume from previously completed checkpoint (e.g., if 10-epoch checkpoint exists, train on additional 10 epochs rather than restart the entire 20 epochs)

- DONT resume checkpoint if training configurations changed
- after resuming, the training losses and other metrics from previous run should still be cocllected in the report
- DONT treat the first epoch after resuming as a genuine start point.

## Repository structure

Please place files (code scripts, logs, artifacts, figures) in dedicated modules.

NEVER stack flat files under the repository root or a module root. Create appropriate modules or submodules (or subfolders).

If a root / module has over 10 files, then you may need to examine whether breaking into subfolders is required. (this is not hard limit, there are situations where large number of files are required, e.g., in a data root with dozens of mat files belonging to different recordings, no appropriate partition exists)

DONT attempt the other extreme -- creating too many folders and only place one file under each folder. 

## Evaluation and performance metrics

NEVER interpret a raw mean performance difference as genuine improvement.

Raw performance metric differences (effect sizes) and statistical tests should BOTH be reported.

Statistical tests MUST retain the magnitude of paired metric differences. Rank-based and sign-only tests (including Wilcoxon and the sign test) are NOT permitted. Use the raw paired mean difference as the test statistic and report its effect size and confidence interval.

- perform test on paired train-test splits, NOT the random seed repeats

AVOID Holm-correction across multiple pairs of model comparisons or task comparisons. Because these are independent comparisons, and may create a p-value measure whose theoretical minimal value is already large. Applying Holm-correction across different metrics within each task is reasonable.

Summarise all metrics for each task, NEVER use a single "primary metric". But you should still emit per-task summary by aggregating the results across metrics.

Undefined correlation metric (e.g., flat signal) can be treated as 0.

NEVER allow the comparison scale or metric definition to depend on the model pool. In that case, comparing another group of models lead to different result interpretations, and this should NEVER happen.

## Artifact saving and logging

The `logs` folder in this repository should ONLY be used for record launch status, logs, failure records, etc. The disk space usage of `logs` should be kept minimal. The actual model fits, checkpoints, result figures, tables, should be saved in a separate artifact root specified by the user. If you are not sure about the artifact and log structure, always ASK ME rather than guesswork.

- NEVER save log files directly under `logs`, add a compaign root
- NEVER save flat log files directly under the compaign's log root that floods the root folder, use subfolders and clean file tree structure!!

Logs should ONLY contain low verbose level basic training cycle information. Reserve detailed logs for replaying failed experiments or smoke tests.

In saved artifacts, reusable model fits and checkpoints should be placed in separate folders (identified by the run hash) rather than saving them under each compaign / experiment's root. This is to ensure that a new compaign including some already fitted models does NOT restart repeated experiments on the already completed model fits.

Within the artifact root of a compaign, you should NOT save the same information twice. e.g., write new results in a new file with previous results copied.

All compaigns / experiments SHOULD share the SAME artifact / log saving logics and modules, rather than each one having its own artifact structure. Minor changes like file addition and field changes are allowed, but NEVER start a completely new artifact or log structure.

DONT save heavy artifacts, save disk-space! This can be done by avoiding saving repeated information. You DONT need to try to keep all records of the entire training process.

- DONT save model checkpoints at every step. Previous checkpoints should be overwritten by latest checkpoints if not being used later. 
- DONT save all model checkpoints for HPO, and never removes them after hyper-parameter is already selected and fixed. These checkpoints should be removed AS SOON AS the hyper-parameter is selected
- DONT save large tensors unless it is necessary for later models. (for example, some classification logits may need to be saved to allow computation of other metrics)

## Plotting

- Font sizes must be LARGE and READABLE. This includes ALL texts on the figure (legend, axis ticks, title, etc.) You can use relatively smaller font sizes for minor labels like axis ticks, but they must still remain READABLE.
- DONT leave large space between title and the main figure. But DONT overlap them.
- use `seaborn` whenever possible rather than `matplotib`
- Scatter plots: set `set_xlim` and (or) `set_ylim` to focus on the main data range.
- Use lines thicker than default, but not too thick that distracts the main data.
- Use `bbox_to_anchor` and `loc'`  to ensure legends are not overlapping with the main figure, not going beyond figure. Legends should have no frame: `frameon=False`.
- Figures should be coloured, but figure legends and labels should make sense even in black-and-white prints. (for example, use different line types or hatches)
- Titles must be wrapped by `textwrap` with suitable width to prevent the title from exceeding boundaries, and title should be kept away from the main figure with at least one line's gap.
- NEVER name title by "Model A vs Model B vs Model C vs ...", use more  scientific names like "Effect of normalisation on CNN", "Comparison between models on high-gamma envelopes"
- Print a message after saving a figure, pointing to its file path. If a group of images are plotted, please ONLY print ONE message pointing to their common folder. compact
- Proper units should be added to the x and y axes if applicable. (DONT add units if no unit)

When the figure components have some semantic relationships, e.g., two lines share the same input, while other 2 uses another input type; or the lines has a clear order (e.g., hidden size increases from 1 to 16), you should assign appropriate line / point types and styles (shape, colour, etc.) to reflect these relations rather than randomly assigning styles. Baseline components (e.g., input-independent baseline, naive model, constant prediction) should receive gray colour.

While the plot layout may change across compaigns / experiments (e.g., different axis definitions, different number of panels, additional shaded areas, change of line style), the overall figure style (font size, legend position, colours used) should be consistent unless explicitly allowed to change. To that said -- all compaign's plotting codes should inherit from shared plotting modules rather than writing completely fresh modules.

If all elements on the plot shares a common setting, it should NOT be included in the legend of every element. (e.g., all have normalisation) If the setting is important (e.g., the input feature set), include in the title, NOT in the legend.

## Constraints

NEVER directly remove or modify a previous completed run result. Appending to an incomplete result is allowed, but NOT modifying previous rows.

- But modifying comparison figures and tables are allowed.

NEVER reveal local paths (e.g., model checkpoints, dataset paths) in files tracked by GitHub.

NEVER hardcode fit-specific (model-specific) labels in scripts for plotting or comparisons. They should be saved in local manifest or global configuration for frequently used experiment results. (e.g., input-independent baseline)

## Documentation

DO NOT overwhelm documentations with heavy technial details. Especially top-level documentations should be brief, and point users to detailed documentations to the corresponding modules.

DO NOT document about the development cycles. If an initial implementation A is changed to B, DONT document "B is implemented, rather than A". You should ONLY document the current version (B) of the code.

- similarly, you should NOT save and commit diagnostics audits, smoke test results, temporary fixes etc.
- If you are not sure, ASK ME rather than assuming a new feature is supposed to be added to this repository as permenant.
