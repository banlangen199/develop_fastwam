from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from fastwam.trainer import Wan22Trainer


def _build_loader(*, pin_memory: bool):
    trainer = object.__new__(Wan22Trainer)
    trainer.seed = 7
    trainer.batch_size = 2
    trainer.num_workers = 0
    trainer.pin_memory = pin_memory
    trainer.accelerator = SimpleNamespace(num_processes=1)
    dataset = TensorDataset(torch.arange(8))
    return trainer._build_loader(dataset)


def test_dataloader_pin_memory_is_disabled_when_configured_off():
    loader = _build_loader(pin_memory=False)
    assert loader.pin_memory is False


def test_dataloader_pin_memory_remains_available_as_opt_in():
    loader = _build_loader(pin_memory=True)
    assert loader.pin_memory is True
