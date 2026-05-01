from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from longitumor.data import labels_to_channels, random_patch_slices
from longitumor.models import LongiTumorMamba, LongiTumorMambaConfig
from longitumor.training import DiceBCELoss


def test_labels_to_channels() -> None:
    labels = torch.tensor([[[1, 2], [3, 4]]]).numpy()
    channels = labels_to_channels(labels)
    assert channels.shape == (4, 1, 2, 2)
    assert channels.sum() == 4


def test_random_patch_slices() -> None:
    slices = random_patch_slices((20, 30, 40), (8, 12, 16))
    assert len(slices) == 3
    assert all((sl.stop or 0) - (sl.start or 0) <= size for sl, size in zip(slices, (8, 12, 16)))


def test_longitumor_forward_backward() -> None:
    model = LongiTumorMamba(LongiTumorMambaConfig(base_channels=4, embedding_dim=16, use_mamba=False))
    x = torch.randn(1, 2, 4, 16, 16, 16)
    availability = torch.ones(1, 2, 4)
    target = torch.rand(1, 2, 4, 16, 16, 16).round()
    output = model(x, availability=availability)
    assert output.logits.shape == target.shape
    assert output.trajectory_embedding.shape == (1, 2, 16)
    loss = DiceBCELoss()(output.logits, target)
    loss.backward()
    assert torch.isfinite(loss)
