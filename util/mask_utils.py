"""Mask (frames, frequencies) spectra without changing sampling policy.

The default return is ``(masked_data, mask_label)``. With
``return_provenance=True``, a third dictionary contains uint8 arrays of the
same shape: ``selection`` (1 temporal, 2 spectral, 3 both) and ``action``
(0 no replacement, 1 zero fill, 2 random replacement). Actions identify the
last write, not equality of input and target. A no-op preserves prior writes.
"""

import random

import numpy as np
import torch

TEMPORAL = 1
SPECTRAL = 2
ZERO = 1
RANDOM = 2


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
        if dice < 0.1:  # TODO look at attentions
            pass
        elif dice < 0.2:
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
        if dice < 0.1:  # TODO look at attentions
            pass
        elif dice < 0.2:
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


def variable_mask_time(
    data: torch.Tensor, task_cfg: object, return_provenance: bool = False
) -> tuple:
    min_size_in_samples = random.randint(1, 2)

    fs = np.linspace(task_cfg.min_f, task_cfg.max_f, task_cfg.n_freq_steps)
    window_sizes = [int(max(min_size_in_samples, 200 / (25 + f))) for f in fs]
    # window_sizes = [int(max(0,160/(30+f))) for f in fs]

    max_size = max(window_sizes)
    valid_starts = list(
        np.arange(max_size, data.shape[0] - max_size)
    )  # remember that mask is centered on time position

    def fill_in_time_mask(array, position, value=None, value_slice=None):
        # value -- what value to fill the template with
        assert not (value != None and value_slice != None)
        arr_len = array.shape[0]
        if value_slice is not None:
            for i in range(len(window_sizes)):
                array[
                    max(0, position - window_sizes[i]) : min(
                        arr_len, position + window_sizes[i]
                    ),
                    i,
                ] = value_slice[i]
        else:
            for i in range(len(window_sizes)):
                array[
                    max(0, position - window_sizes[i]) : min(
                        arr_len, position + window_sizes[i]
                    ),
                    i,
                ] = value
        return array

    def take_time_mask(array, position):
        arr_len = array.shape[0]
        value_slice = []
        for i in range(len(window_sizes)):
            value_slice.append(
                array[
                    max(0, position - window_sizes[i]) : min(
                        arr_len, position + window_sizes[i]
                    ),
                    i,
                ]
            )
        return value_slice

    masked_positions = []
    max_window = 2 * max(window_sizes)
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
    mask_label = torch.zeros_like(data)
    action = torch.zeros_like(data, dtype=torch.uint8)
    for position in masked_positions:
        dice = random.random()
        if dice < 0.1:  # TODO look at attentions
            pass
        elif dice < 0.2:
            random_position = valid_starts[
                random.randint(0, len(valid_starts) - 1)
            ]
            value_slice = take_time_mask(data, random_position)
            masked_data = fill_in_time_mask(
                masked_data, position, value_slice=value_slice
            )
            fill_in_time_mask(action, position, value=RANDOM)
        else:
            mask_fill_value = get_mask_fill_value(data)
            masked_data = fill_in_time_mask(
                masked_data, position, value=mask_fill_value
            )
            fill_in_time_mask(action, position, value=ZERO)
        mask_label = fill_in_time_mask(mask_label, position, 1)
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
    mask_label = torch.zeros_like(data)
    action = torch.zeros_like(data, dtype=torch.uint8)

    mask_fill_value = get_mask_fill_value(data)
    for position in masked_positions:
        diff = idx2mask_size[position]
        dice = random.random()
        if dice < 0.1:  # TODO look at attentions
            pass
        elif dice < 0.2:
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
