"""Mask (frames, frequencies) spectra without changing sampling policy.

The default return is ``(masked_data, mask_label)``. With
``return_provenance=True``, a third dictionary contains uint8 arrays of the
same shape: ``selection`` (1 temporal, 2 spectral, 3 both) and ``action``
(0 no replacement, 1 zero fill, 2 random replacement). Actions identify the
last write, not equality of input and target. A no-op preserves prior writes.
"""

import random
import copy
import math
from functools import lru_cache

from omegaconf import OmegaConf, open_dict
from torch.utils.data import Subset

import numpy as np
import torch

MASK_TEMPLATE_CACHE_SIZE = 16
TEMPORAL = 1
SPECTRAL = 2
ZERO = 1
RANDOM = 2


def replacement_thresholds(policy: object = None) -> tuple[float, float]:
    """Validate replacement probabilities and return cumulative thresholds.

    Parameters
    ----------
    policy : mapping or None, optional
        Zero, random, and unchanged probabilities; None uses original 80/10/10.
    """
    if policy is None:
        return 0.1, 0.2
    if set(policy) != {"zero", "random", "unchanged"}:
        raise ValueError("Replacement policy needs zero, random, unchanged.")
    values = [float(policy[k]) for k in ("zero", "random", "unchanged")]
    if any(not math.isfinite(p) or p < 0 for p in values):
        raise ValueError(
            "Replacement probabilities must be finite and nonnegative."
        )
    if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
        raise ValueError("Replacement probabilities must sum to one.")
    return values[2], values[2] + values[1]


def test_masked_dataset(
    dataset: object, policy: object = None, targets_only: bool = False,
) -> object:
    """Clone dataset configuration for test-only masking without mutating training.

    Parameters
    ----------
    dataset : Dataset
        Masked dataset, optionally nested in Subsets; sample indices stay fixed.
    policy : mapping or None, optional
        Explicit replacement probabilities; default None.
    targets_only : bool, optional
        Skip unused stochastic masks in diagnostic copies, default False.
    """
    if policy is None and not targets_only:
        return dataset
    if policy is not None:
        replacement_thresholds(policy)
    if isinstance(dataset, Subset):
        return Subset(
            test_masked_dataset(dataset.dataset, policy, targets_only),
            dataset.indices
        )
    if not hasattr(dataset, "task_cfg"):
        raise ValueError("Test masking requires a dataset with task_cfg.")
    cloned = copy.copy(dataset)
    cloned.task_cfg = copy.deepcopy(dataset.task_cfg)
    with open_dict(cloned.task_cfg):
        if policy is not None:
            cloned.task_cfg.replacement_probabilities = OmegaConf.create(
                dict(policy)
            )
        if targets_only:
            cloned.task_cfg.diagnostic_targets_only = True
    return cloned


def get_mask_fill_value(data):
    return 0
    # return data.min() - 1


def create_masked_intervals(
    data, consecutive_min, consecutive_max, mask_p, axis="time"
):
    assert consecutive_min <= consecutive_max
    assert consecutive_max < data.shape[0]

    if axis == "time":
        valid_starts = range(len(data) - consecutive_max)
    else:
        valid_starts = range(data.shape[1] - consecutive_max)
    masked_steps = []
    for i in valid_starts:
        if random.random() < mask_p:
            if len(masked_steps) == 0 or masked_steps[-1][1] < i:
                masked_steps.append(
                    (i, i + random.randint(consecutive_min, consecutive_max))
                )
    # masked_steps = [i for i in valid_starts if random.random() < mask_p]
    # masked_steps = [(i, i+random.randint(consecutive_min, consecutive_max)) for i in masked_steps]
    return masked_steps, valid_starts


def fixed_mask_inputs(
    data: torch.Tensor, task_cfg: object, return_provenance: bool = False
) -> tuple:
    identity, random_end = replacement_thresholds(
        task_cfg.get("replacement_probabilities")
    )
    mask_label = torch.zeros_like(data)
    action = torch.zeros_like(data, dtype=torch.uint8)

    masked_steps, valid_starts = create_masked_intervals(
        data,
        task_cfg.time_mask_consecutive_min,
        task_cfg.time_mask_consecutive_max,
        task_cfg.time_mask_p,
        axis="time",
    )
    selection = torch.zeros_like(data, dtype=torch.uint8)
    for start, end in masked_steps:
        mask_label[start:end, :] = 1
        selection[start:end, :] |= TEMPORAL

    masked_data = torch.clone(data)
    mask_fill_value = get_mask_fill_value(data)
    for start, end in masked_steps:
        dice = random.random()
        if dice < identity:  # TODO look at attentions
            pass
        elif dice < random_end:
            random_replace_start = random.randint(0, len(valid_starts) - 1)
            diff = end - start
            masked_data[start:end, :] = data[
                random_replace_start : random_replace_start + diff, :
            ]
            action[start:end, :] = RANDOM
        else:
            masked_data[start:end, :] = mask_fill_value
            action[start:end, :] = ZERO

    masked_steps, valid_starts = create_masked_intervals(
        data,
        task_cfg.freq_mask_consecutive_min,
        task_cfg.freq_mask_consecutive_max,
        task_cfg.freq_mask_p,
        axis="freq",
    )
    for start, end in masked_steps:
        mask_label[:, start:end] = 1
        selection[:, start:end] |= SPECTRAL
    for start, end in masked_steps:
        dice = random.random()
        if dice < identity:  # TODO look at attentions
            pass
        elif dice < random_end:
            random_replace_start = valid_starts[
                random.randint(0, len(valid_starts) - 1)
            ]
            diff = end - start
            masked_data[:, start:end] = data[
                :, random_replace_start : random_replace_start + diff
            ]
            action[:, start:end] = RANDOM
        else:
            masked_data[:, start:end] = mask_fill_value
            action[:, start:end] = ZERO
    if return_provenance:
        return (
            masked_data,
            mask_label,
            {
                "selection": selection,
                "action": action,
            },
        )
    return masked_data, mask_label


@lru_cache(maxsize=MASK_TEMPLATE_CACHE_SIZE)
def temporal_mask_template(
    min_f: float, max_f: float, bins: int, minimum: int,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Build relative frame and bin indices for a single temporal mask.

    Returns the largest half-width and two integer vectors of length N,
    one entry per selected cell. Callers offset frames by the mask center.
    """
    frequencies = np.linspace(min_f, max_f, bins)
    widths = [int(max(minimum, 200 / (25 + f))) for f in frequencies]
    offsets = np.concatenate([np.arange(-w, w) for w in widths])
    channels = np.repeat(np.arange(bins), np.multiply(2, widths))
    return max(widths), torch.from_numpy(offsets), torch.from_numpy(channels)


def variable_mask_time(
    data: torch.Tensor, task_cfg: object, return_provenance: bool = False
) -> tuple:
    min_size_in_samples = random.randint(1, 2)

    max_size, offsets, channels = temporal_mask_template(
        task_cfg.min_f, task_cfg.max_f, task_cfg.n_freq_steps,
        min_size_in_samples,
    )
    valid_starts = range(max_size, data.shape[0] - max_size)

    masked_positions = []
    max_window = 2 * max_size
    mask_probability = task_cfg.mask_p
    for pos in valid_starts:
        if random.random() < mask_probability:
            if (
                len(masked_positions) == 0
                or abs(masked_positions[-1] - pos) > max_window + 1
            ):
                masked_positions.append(pos)
    # import pdb; pdb.set_trace()
    masked_data = torch.clone(data)
    identity, random_end = replacement_thresholds(
        task_cfg.get("replacement_probabilities")
    )
    mask_label = torch.zeros_like(data)
    action = torch.zeros_like(data, dtype=torch.uint8)
    for position in masked_positions:
        rows = offsets + position
        dice = random.random()
        if dice >= identity:
            if dice < random_end:
                donor = valid_starts[random.randint(0, len(valid_starts) - 1)]
                masked_data[rows, channels] = data[offsets + donor, channels]
                action[rows, channels] = RANDOM
            else:
                masked_data[rows, channels] = get_mask_fill_value(data)
                action[rows, channels] = ZERO
        mask_label[rows, channels] = 1
    if return_provenance:
        return (
            masked_data,
            mask_label,
            {
                "selection": mask_label.to(torch.uint8),
                "action": action,
            },
        )
    return masked_data, mask_label


def variable_mask_freq(
    data: torch.Tensor, task_cfg: object, return_provenance: bool = False
) -> tuple:
    fs = np.linspace(task_cfg.min_f, task_cfg.max_f, task_cfg.n_freq_steps)
    # mask_sizes = list(reversed([max(1,int(max(0,160/(30+f)))) for f in fs]))
    mask_sizes = [max(1, int(4.9 * (f) / 250)) for f in fs]
    idx2mask_size = {i: s for i, s in enumerate(mask_sizes)}
    valid_starts = list(range(data.shape[1] - max(mask_sizes)))
    mask_probability = task_cfg.mask_p
    masked_positions = [
        i for i in valid_starts if random.random() < mask_probability
    ]

    masked_data = torch.clone(data)
    identity, random_end = replacement_thresholds(
        task_cfg.get("replacement_probabilities")
    )
    mask_label = torch.zeros_like(data)
    action = torch.zeros_like(data, dtype=torch.uint8)

    mask_fill_value = get_mask_fill_value(data)
    for position in masked_positions:
        diff = idx2mask_size[position]
        dice = random.random()
        if dice < identity:  # TODO look at attentions
            pass
        elif dice < random_end:
            random_replace_start = valid_starts[
                random.randint(0, len(valid_starts) - 1)
            ]
            masked_data[:, position : position + diff] = data[
                :, random_replace_start : random_replace_start + diff
            ]
            action[:, position : position + diff] = RANDOM
        else:
            masked_data[:, position : position + diff] = mask_fill_value
            action[:, position : position + diff] = ZERO
        mask_label[:, position : position + diff] = 1
    if return_provenance:
        return (
            masked_data,
            mask_label,
            {
                "selection": mask_label.to(torch.uint8) * SPECTRAL,
                "action": action,
            },
        )
    return masked_data, mask_label


def variable_mask(
    data: torch.Tensor, task_cfg: object, return_provenance: bool = False
) -> tuple:
    masked_data, mask_label, temporal = variable_mask_time(
        data, task_cfg, return_provenance=True
    )
    masked_data, freq_mask_label, spectral = variable_mask_freq(
        masked_data, task_cfg, return_provenance=True
    )
    mask_label += freq_mask_label
    if return_provenance:
        return (
            masked_data,
            mask_label,
            {
                "selection": temporal["selection"] | spectral["selection"],
                "action": torch.where(
                    spectral["action"] != 0,
                    spectral["action"],
                    temporal["action"],
                ),
            },
        )
    return masked_data, mask_label


def mask_inputs(
    data: torch.Tensor, task_cfg: object, return_provenance: bool = False
) -> tuple:
    """Return masked spectra, loss selection, and optional write provenance.

    Parameters
    ----------
    data : torch.Tensor
        Input spectrum of shape (frames, frequencies).
    task_cfg : DictConfig
        Existing fixed or variable masking settings.
    return_provenance : bool, optional
        Include uint8 selection/action maps; default False.

    Returns
    -------
    tuple
        Two spectrum-shaped tensors and optionally a provenance dictionary.
    """
    functions = {"fixed": fixed_mask_inputs, "variable": variable_mask}
    if task_cfg.mask_type not in functions:
        raise ValueError(f"Unknown mask type: {task_cfg.mask_type!r}")
    return functions[task_cfg.mask_type](
        data, task_cfg, return_provenance=return_provenance
    )
