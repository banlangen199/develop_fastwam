from pathlib import Path

import pytest

from fastwam.trainer import Wan22Trainer


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def load_checkpoint(self, path, optimizer=None):
        self.calls.append((path, optimizer))


def _bare_trainer(model, resume):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.resume = resume
    trainer._weight_checkpoint_loaded_before_optimizer = False
    return trainer


def test_weight_checkpoint_is_preloaded_once(tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()
    model = _RecordingModel()
    trainer = _bare_trainer(model, checkpoint)

    trainer._load_weight_checkpoint_before_optimizer()

    assert model.calls == [(str(checkpoint), None)]
    assert trainer._weight_checkpoint_loaded_before_optimizer is True

    # Post-prepare resume handling must not load the same file a second time.
    trainer._resume_or_load_checkpoint()
    assert model.calls == [(str(checkpoint), None)]


def test_full_state_directory_is_not_preloaded(tmp_path: Path):
    state_dir = tmp_path / "step_000123"
    state_dir.mkdir()
    model = _RecordingModel()
    trainer = _bare_trainer(model, state_dir)

    trainer._load_weight_checkpoint_before_optimizer()

    assert model.calls == []
    assert trainer._weight_checkpoint_loaded_before_optimizer is False


def test_missing_weight_checkpoint_fails_before_optimizer(tmp_path: Path):
    missing = tmp_path / "missing.pt"
    trainer = _bare_trainer(_RecordingModel(), missing)

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        trainer._load_weight_checkpoint_before_optimizer()
