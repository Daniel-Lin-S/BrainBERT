"""Check optional step decay, linear warmup and checkpoint restoration."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from schedulers.ramp_up import DECAY_GAMMA, DECAY_INTERVALS, RampUp

BASE_LR = 1e-4
TOTAL_STEPS = 1050
WARMUP_FRACTION = 0.13


def make_schedule(decay=None):
    """Build a scalar CPU optimiser with an optional configuration flag."""
    options = dict(total_steps=TOTAL_STEPS, warmup=WARMUP_FRACTION)
    if decay is not None:
        options['decay'] = decay
    optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    return optim, RampUp(SimpleNamespace(**options), optim)


def advance(optim, schedule, steps):
    """Return learning rates after the requested optimiser updates."""
    rates = []
    for _ in range(steps):
        optim.step()
        schedule.step()
        rates.append(schedule.get_lr())
        assert rates[-1] == optim.param_groups[0]['lr']
    return rates


@pytest.mark.parametrize('decay', [None, False, True])
def test_warmup_and_optional_decay(decay):
    """Fractional historical intervals become effective integer intervals."""
    optim, schedule = make_schedule(decay)
    warmup = int(TOTAL_STEPS * WARMUP_FRACTION)
    assert schedule.get_lr() == 0
    rates = advance(optim, schedule, TOTAL_STEPS)
    for step in (1, warmup // 2, warmup):
        assert rates[step - 1] == pytest.approx(BASE_LR * step / warmup)
    if decay:
        interval = math.ceil((TOTAL_STEPS - warmup) / DECAY_INTERVALS)
        assert type(schedule.scheduler.after_scheduler.step_size) is int
        assert rates[warmup + interval] == pytest.approx(BASE_LR * DECAY_GAMMA)
        reductions = (TOTAL_STEPS - warmup - 1) // interval
        assert rates[-1] == pytest.approx(BASE_LR * DECAY_GAMMA ** reductions)
    else:
        assert rates[warmup:] == pytest.approx([BASE_LR] * (TOTAL_STEPS-warmup))
        assert schedule.scheduler.after_scheduler is None


@pytest.mark.parametrize('decay', [False, True])
@pytest.mark.parametrize('stop', [30, 450])
def test_scheduler_resume_matches_uninterrupted(decay, stop):
    """Restore warmup or decay without restarting either phase."""
    optim, schedule = make_schedule(decay)
    advance(optim, schedule, stop)
    optim_state = copy.deepcopy(optim.state_dict())
    state = copy.deepcopy(schedule.get_state_dict())
    expected = advance(optim, schedule, TOTAL_STEPS - stop)
    resumed_optim, resumed = make_schedule(decay)
    resumed_optim.load_state_dict(optim_state)
    resumed.load_state_dict(state)
    assert advance(resumed_optim, resumed, TOTAL_STEPS-stop) == expected


@pytest.mark.parametrize('options, message', [
    ({'decay': 'false'}, 'decay must be a boolean'),
    ({'total_steps': 1}, 'total_steps must be an integer'),
    ({'warmup': float('nan')}, 'warmup must be a finite fraction'),
    ({'warmup': 0}, 'warmup must be a finite fraction'),
    ({'warmup': .00001}, 'at least one update'),
])
def test_invalid_schedule_fails_clearly(options, message):
    config = dict(total_steps=TOTAL_STEPS, warmup=WARMUP_FRACTION)
    config.update(options)
    optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    with pytest.raises(ValueError, match=message):
        RampUp(SimpleNamespace(**config), optim)
