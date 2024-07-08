import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from loguru import logger

# logger.add(sys.stdout, format="[{time}] [<level>{level}</level>] {message}")
# logger.level("INFO", color="<red>")


class MultiModelMemoryTrainer:
    """
    A class that manages multiple PyTorch models and dynamically allocates
    GPU memory based on their learning speed.
    """

    def __init__(
        self,
        models: List[nn.Module],
        initial_memory_allocation: List[float],
        total_memory: int,
    ):
        """
        Initialize the MultiModelMemoryTrainer class.

        Args:
            models (List[nn.Module]): A list of PyTorch models to manage.
            initial_memory_allocation (List[float]): Initial memory allocation ratios for each model (should sum to 1.0).
            total_memory (int): Total GPU memory to allocate across all models (in bytes).
        """
        self.models = models
        self.memory_allocation = initial_memory_allocation
        self.total_memory = total_memory
        self.learning_rates = [0.0] * len(models)
        self.optimizers = [
            optim.Adam(model.parameters()) for model in models
        ]

        logger.info(
            f"Initialized MultiModelMemoryTrainer with {len(models)} models"
        )

        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This class requires GPU support."
            )

        self._apply_memory_allocation()

    def update_learning_rates(
        self, new_learning_rates: List[float]
    ) -> None:
        """
        Update the learning rates for each model.

        Args:
            new_learning_rates (List[float]): New learning rates for each model.
        """
        self.learning_rates = new_learning_rates
        logger.debug(f"Updated learning rates: {self.learning_rates}")

    def reallocate_gpu_memory(self) -> None:
        """
        Reallocate GPU memory based on the current learning rates.
        """
        total_learning_rate = sum(self.learning_rates)

        if total_learning_rate == 0:
            logger.warning(
                "Total learning rate is zero. Skipping reallocation."
            )
            return

        new_allocation = [
            rate / total_learning_rate for rate in self.learning_rates
        ]

        logger.info(f"New GPU memory allocation: {new_allocation}")
        self.memory_allocation = new_allocation

        self._apply_memory_allocation()

    def _apply_memory_allocation(self) -> None:
        """
        Apply the current GPU memory allocation to the models.
        """
        torch.cuda.empty_cache()  # Clear any allocated memory

        for i, (model, allocation) in enumerate(
            zip(self.models, self.memory_allocation)
        ):
            memory_to_allocate = int(self.total_memory * allocation)
            logger.debug(
                f"Allocating {memory_to_allocate} bytes of GPU memory to model {i}"
            )

            # Move model to CPU first
            model.cpu()

            # Allocate memory on GPU
            torch.cuda.set_per_process_memory_fraction(allocation)
            torch.cuda.empty_cache()

            # Move model back to GPU
            model.cuda()

            # Verify allocation
            memory_allocated = torch.cuda.memory_allocated()
            logger.debug(
                f"Actually allocated {memory_allocated} bytes for model {i}"
            )

    def train_step(
        self, data: Dict[str, torch.Tensor]
    ) -> List[float]:
        """
        Perform a training step for all models.

        Args:
            data (Dict[str, torch.Tensor]): Training data for all models.

        Returns:
            List[float]: List of losses for each model after the training step.
        """
        losses = []
        for i, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers)
        ):
            loss = self._train_model(model, optimizer, data)
            losses.append(loss)

        logger.info(f"Training step completed. Losses: {losses}")
        return losses

    def _train_model(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        data: Dict[str, torch.Tensor],
    ) -> float:
        """
        Train an individual model.

        Args:
            model (nn.Module): The model to train.
            optimizer (optim.Optimizer): The optimizer for the model.
            data (Dict[str, torch.Tensor]): Training data for the model.

        Returns:
            float: The loss after training.
        """
        model.train()
        optimizer.zero_grad()

        inputs = data["inputs"].cuda()
        targets = data["targets"].cuda()

        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        return loss.item()

    def validate(self, data: Dict[str, torch.Tensor]) -> List[float]:
        """
        Validate all models.

        Args:
            data (Dict[str, torch.Tensor]): Validation data for all models.

        Returns:
            List[float]: List of validation losses for each model.
        """
        val_losses = []
        for model in self.models:
            val_loss = self._validate_model(model, data)
            val_losses.append(val_loss)

        logger.info(f"Validation completed. Losses: {val_losses}")
        return val_losses

    def _validate_model(
        self, model: nn.Module, data: Dict[str, torch.Tensor]
    ) -> float:
        """
        Validate an individual model.

        Args:
            model (nn.Module): The model to validate.
            data (Dict[str, torch.Tensor]): Validation data for the model.

        Returns:
            float: The validation loss.
        """
        model.eval()
        with torch.no_grad():
            inputs = data["inputs"].cuda()
            targets = data["targets"].cuda()

            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)

        return loss.item()
