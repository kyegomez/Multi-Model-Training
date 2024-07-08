import pytest
import torch
import torch.nn as nn
from multi_model_trainers.main import MultiModelMemoryTrainer


@pytest.fixture
def gpu_allocator():
    models = [
        nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
        for _ in range(3)
    ]
    initial_allocation = [1 / 3, 1 / 3, 1 / 3]
    total_memory = 4 * 1024 * 1024 * 1024  # 4 GB
    return MultiModelMemoryTrainer(
        models, initial_allocation, total_memory
    )


def test_train_step(gpu_allocator):
    train_data = {
        "inputs": torch.rand(32, 10),
        "targets": torch.rand(32, 1),
    }
    losses = gpu_allocator.train_step(train_data)
    assert len(losses) == len(gpu_allocator.models)


def test_update_learning_rates(gpu_allocator):
    losses = [1.0, 2.0, 3.0]
    gpu_allocator.update_learning_rates(losses)
    for rate, model in zip(
        gpu_allocator.learning_rates, gpu_allocator.models
    ):
        assert rate == model.lr


def test_reallocate_gpu_memory(gpu_allocator):
    gpu_allocator.reallocate_gpu_memory()
    assert (
        sum(gpu_allocator.memory_allocation)
        == gpu_allocator.total_memory
    )


def test_validate(gpu_allocator):
    val_data = {
        "inputs": torch.rand(64, 10),
        "targets": torch.rand(64, 1),
    }
    val_losses = gpu_allocator.validate(val_data)
    assert len(val_losses) == len(gpu_allocator.models)
