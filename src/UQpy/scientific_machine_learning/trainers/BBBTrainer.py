import torch
import torch.nn as nn
import logging
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveInteger


# @beartype
class BBBTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module = nn.MSELoss(),
    ):
        """Prepare to train a Bayesian neural network using Bayes by back propagation

        :param model: Bayesian Neural Network model to be trained
        :param optimizer: Optimization algorithm used to update ``model`` parameters
        :param loss_function: Function used to compute negative log likelihood of the data during training
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.history: dict = {
            "train_loss": None,
            "train_kl": None,
            "train_nll": None,
            "test_nll": None,
        }
        """Record of the loss during training and validation. 
        Note if training ends early there may be ``NaN`` values, as the histories are initialized with ``NaN``.
        
        - ``history["train_loss"]`` contains the training history as a ``torch.Tensor``.
        - ``history["train_kl"]`` contains the training Kullback–Leibler divergence as a ``torch.Tensor``.
        - ``history["train_nll"]`` contains the training negative log likelihood loss as a ``torch.Tensor``.
        - ``history["test_nll"]`` contains the testing negative log likelihood loss as a ``torch.Tensor``.
         """
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        train_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
        epochs: PositiveInteger = 100,
        num_samples: PositiveInteger = 1,
        tolerance: float = 1e-3,
        beta: float = 1.0,
    ):
        """Run the ``optimizer`` algorithm to learn the parameters of the ``model`` that fit ``train_data``

        :param train_data: Data used to compute ``model`` loss
        :param test_data: Data used to validate the performance of the model
        :param epochs: Maximum number of epochs to run the ``optimizer`` for
        :param num_samples: Number of Monte Carlo samples to approximate the loss
        :param tolerance: Optimization terminates early if *average* training loss is below tolerance.
         Default :math:`1e-3`
         :param beta: Weighting for the KL term in ELBO loss. Default :math: '1.0'

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
                [epochs], torch.nan, requires_grad=False
            )
            self.history["train_kl"] = torch.full(
                [epochs], torch.nan, requires_grad=False
            )
            self.history["train_nll"] = torch.full(
                [epochs], torch.nan, requires_grad=False
            )
        if test_data:
            self.history["test_nll"] = torch.full(
                [epochs], torch.nan, requires_grad=False
            )

        self.logger.info("UQpy: Scientific Machine Learning: Beginning " + log_note)
        i = 0
        average_train_loss = torch.inf
        while i < epochs and average_train_loss > tolerance:
            if train_data:
                self.model.train(True)
                total_train_loss = 0
                total_nll_loss = 0
                total_kl_loss = 0
                for batch_number, (*x, y) in enumerate(train_data):
                    nll_loss = torch.zeros(num_samples)
                    for sample in range(num_samples):
                        prediction = self.model(*x)
                        nll_loss[sample] = self.loss_function(prediction, y)
                    kl_loss = self.model.compute_kullback_leibler_divergence()
                    mean_nll = torch.mean(nll_loss)
                    train_loss = mean_nll + beta * kl_loss
                    train_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_train_loss += train_loss.item()
                    total_nll_loss += mean_nll.item()
                    total_kl_loss += kl_loss
                average_train_loss = total_train_loss / len(train_data)
                self.history["train_loss"][i] = total_train_loss
                self.history["train_nll"][i] = total_nll_loss
                self.history["train_kl"][i] = total_kl_loss
                self.logger.info(
                    f"UQpy: Scientific Machine Learning: "
                    f"Epoch {i + 1:,} / {epochs:,} "
                    f"Train Loss {total_train_loss} "
                    f"Train NLL {total_nll_loss} "
                    f"Train KL {total_kl_loss}"
                )
                self.model.train(False)
            if test_data:
                total_test_nll = 0
                with torch.no_grad():
                    for batch_number, (*x, y) in enumerate(test_data):
                        test_prediction = self.model(*x)
                        test_nll = self.loss_function(test_prediction, y)
                        total_test_nll += test_nll.item()
                    self.history["test_nll"][i] = total_test_nll
                    self.logger.info(
                        f"UQpy: Scientific Machine Learning: "
                        f"Epoch {i + 1:,} / {epochs:,} Test NLL {total_test_nll}"
                    )
            i += 1

        self.logger.info(f"UQpy: Scientific Machine Learning: Completed " + log_note)