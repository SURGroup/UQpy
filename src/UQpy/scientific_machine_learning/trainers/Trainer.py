import torch
import torch.nn as nn
import logging
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveInteger


@beartype
class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module = nn.MSELoss(),
    ):
        """Prepare to train a neural network

        :param model: Neural Network model to be trained
        :param optimizer: Optimization algorithm used to update ``model`` parameters
        :param loss_function: Function used to compute loss during training
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        # ToDo: make sure this can detect bayesian layers and adjust the train method appropriately

        self.history: dict = {"train_loss": None, "test_loss": None}
        """Record of the loss during training and validation. 
         Training history is stored as a ``torch.Tensor`` in ``history["train_loss"]``.
         Testing history is stored as a ``torch.Tensor`` in ``history["test_loss"]``.
         Note if training ends early there may be ``NaN`` values, as the histories are initialized with ``NaN``."""
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        train_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
        epochs: PositiveInteger = 100,
        tolerance: float = 1e-3,
    ):
        """Run the ``optimizer`` algorithm to learn the parameters of the ``model`` that fit ``train_data``

        :param train_data: Data used to compute ``model`` loss
        :param test_data: Data used to validate the performance of the model
        :param epochs: Maximum number of epochs to run the ``optimizer`` for
        :param tolerance: Optimization terminates early if *average* training loss is below tolerance.
         Default :math:`1e-3`

        :raises RuntimeError: If neither ``train_data`` nor ``test_data`` is provided, a RuntimeError occurs.
        """
        if train_data and not test_data:
            log_note = f"training {self.model.__class__.__name__}"
        elif not train_data and test_data:
            log_note = f"testing {self.model.__class__.__name__}"
        elif train_data and test_data:
            log_note = f"training and testing {self.model.__class__.__name__}"
        else:
            raise RuntimeError(
                "UQpy: At least one of `train_data` or `test_data` must be provided."
            )

        if train_data:
            self.history["train_loss"] = torch.full(
                epochs, torch.nan, requires_grad=False
            )
        if test_data:
            self.history["test_loss"] = torch.full(
                epochs, torch.nan, requires_grad=False
            )

        self.logger.info("UQpy: Scientific Machine Learning: Beginning " + log_note)
        i = 0
        average_train_loss = torch.inf
        while i < epochs and average_train_loss > tolerance:
            if train_data:
                self.model.train(True)
                total_train_loss = 0
                for batch_number, (x, y) in enumerate(train_data):
                    prediction = self.model(x)
                    train_loss = self.loss_function(prediction, y)
                    train_loss.backward()
                    total_train_loss += train_loss.item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                average_train_loss = total_train_loss / len(train_data)
                self.history["train_loss"][i] = total_train_loss
                self.logger.info(
                    f"UQpy: Scientific Machine Learning: "
                    f"Epoch {i+1:,} / {epochs:,} Train Loss {total_train_loss}"
                )
                self.model.train(False)
            if test_data:
                total_test_loss = 0
                with torch.no_grad():
                    for batch_number, (x, y) in enumerate(test_data):
                        test_prediction = self.model(x)
                        test_loss = self.loss_function(test_prediction, y)
                        total_test_loss += test_loss.item()
                    self.history["test_loss"][i] = total_test_loss
                    self.logger.info(
                        f"UQpy: Scientific Machine Learning: "
                        f"Epoch {i+1:,} / {epochs:,} Test Loss {total_test_loss}"
                    )
            i += 1

        self.logger.info(f"UQpy: Scientific Machine Learning: Completed " + log_note)