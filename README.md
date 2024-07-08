[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Multi-Model Trainers
Over the past year, Agora has implemented and created thousands of models all with slightly different variations and architectures. So a question arose, which was: How can we train multiple models all at once and then evaluate them and then distribute more GPU memory to the models that are learning the fastest with the lowest loss. There are many configurations we'll attempt in the future, but this is super experimental! If you want to hack on this, Join us at Agora and let's accelerate!

## Install
```bash
$ pip install multi-model-trainers
```

## Usage

```python
import torch
import torch.nn as nn
from loguru import logger

from multi_model_trainers.main import MultiModelMemoryTrainer

# Example usage
if __name__ == "__main__":
    # Create some dummy models
    models = [
        nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
        for _ in range(3)
    ]
    initial_allocation = [1 / 3, 1 / 3, 1 / 3]
    total_memory = 4 * 1024 * 1024 * 1024  # 4 GB

    gpu_allocator = MultiModelMemoryTrainer(
        models, initial_allocation, total_memory
    )

    # Simulate a few training steps
    for step in range(5):
        logger.info(f"Training step {step}")

        # Generate dummy data
        train_data = {
            "inputs": torch.rand(32, 10),
            "targets": torch.rand(32, 1),
        }

        losses = gpu_allocator.train_step(train_data)

        # Update learning rates based on losses (this is a simplistic approach)
        learning_rates = [1 / (loss + 1e-5) for loss in losses]
        gpu_allocator.update_learning_rates(learning_rates)

        # Reallocate GPU memory
        gpu_allocator.reallocate_gpu_memory()

        # Validation step
        val_data = {
            "inputs": torch.rand(64, 10),
            "targets": torch.rand(64, 1),
        }
        val_losses = gpu_allocator.validate(val_data)

    logger.info("Training complete")
```


# License
MIT
