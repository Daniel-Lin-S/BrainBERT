"""Compare accumulated gradients and monitoring against a full batch."""
import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from criterions.pretrain_masked_criterion import PretrainMaskedCriterion


class Reconstruction(torch.nn.Module):
    """Use a deterministic reconstruction for gradient comparisons."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, values, padding):
        return self.linear(values), torch.zeros_like(values)


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("weighted", [False, True])
def test_accumulated_gradients_and_monitoring(size, weighted):
    """Preserve global denominators with unequal masked/content cell counts."""
    torch.manual_seed(7)
    model = Reconstruction()
    accumulated = copy.deepcopy(model)
    target = torch.randn(8, 6, 3) * 3
    target[0] = 0
    mask = torch.rand_like(target) > 0.4
    mask[:, 0, 0] = True
    batch = {
        "target": target,
        "masked_input": torch.randn_like(target),
        "mask_label": mask,
        "attn_mask": torch.zeros(8, 6, dtype=torch.bool),
        "wavs": [None] * 8,
        "mask_provenance": {
            "selection": mask.to(torch.uint8),
            "action": mask.to(torch.uint8),
        },
    }
    criterion = PretrainMaskedCriterion()
    criterion.build_criterion(OmegaConf.create({
        "alpha": 2,
        "content_spectral_weighting": {
            "apply": weighted, "strength": 1.0, "exponent": 2.0,
        },
    }))
    criterion.frequency_hz = np.arange(3, dtype=float)
    loss, expected = criterion(model, batch, "cpu")
    loss.backward()
    actual = criterion.accumulated_backward(
        accumulated, batch, "cpu", size,
    )
    for first, second in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(first.grad, second.grad)
    for key in expected:
        if key == "images":
            continue
        if key == "monitor_statistics":
            for field in expected[key]:
                np.testing.assert_allclose(
                    actual[key][field], expected[key][field], rtol=1e-6,
                    atol=1e-6,
                )
        else:
            assert actual[key] == pytest.approx(expected[key], rel=1e-6)



def test_adaptive_microbatch_requires_sustained_memory(monkeypatch):
    """Respect every device and the allocator cap before increasing rows."""
    from runner import Runner

    runner = Runner.__new__(Runner)
    runner.device = torch.device("cuda", 0)
    runner.microbatch_ready_checks = 0
    runner.microbatch_last_check = -1
    runner.cfg = OmegaConf.create({
        "microbatch_size": 64, "train_batch_size": 256,
        "multi_gpu": True, "gpu_memory_fraction": 0.5,
        "adaptive_microbatch": {
            "check_interval": 100, "consecutive_checks": 3,
            "margin_bytes": 100,
            "profile_peaks_bytes": {"128": 500, "256": 700},
        },
    })
    free = [1000] * 4
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda device: (free[device], 4000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    for step in (100, 200):
        runner.completed_steps = step
        runner.maybe_relax_microbatch()
    runner.maybe_relax_microbatch()
    assert runner.microbatch_ready_checks == 2
    assert runner.cfg.microbatch_size == 64
    free[3] = 400
    runner.completed_steps = 300
    runner.maybe_relax_microbatch()
    assert runner.microbatch_ready_checks == 0
    free[3] = 1000
    for step in (400, 500, 600):
        runner.completed_steps = step
        runner.maybe_relax_microbatch()
    assert runner.cfg.microbatch_size == 128
    runner.cfg.gpu_memory_fraction = 0.1
    for step in (700, 800, 900):
        runner.completed_steps = step
        runner.maybe_relax_microbatch()
    assert runner.cfg.microbatch_size == 128
    runner.cfg.gpu_memory_fraction = 0.5
    for step in (1000, 1100, 1200):
        runner.completed_steps = step
        runner.maybe_relax_microbatch()
    assert runner.cfg.microbatch_size is None
