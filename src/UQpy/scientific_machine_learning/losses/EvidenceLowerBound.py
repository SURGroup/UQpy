import torch
from beartype import beartype
from UQpy.scientific_machine_learning.baseclass.Loss import Loss
from UQpy.utilities.ValidationTypes import PositiveInteger, PositiveFloat


@beartype
class EvidenceLowerBound(Loss):
    def __init__(self, train_size: PositiveInteger, loss_function, **kwargs):
        """

        # ToDo: figure out default loss function and typing

        :param train_size: Number of training samples
        :param loss_function:
        """
        super().__init__(**kwargs)
        self.train_size = train_size
        self.loss_function = loss_function

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        kl: PositiveFloat,
        kl_weight: PositiveFloat,
    ) -> torch.Tensor:
        """

        :param x: Predicted values
        :param target: Target values
        :param kl: Kullback-Leibler Divergence between variational posterior and prior
        :param kl_weight: Weighting factor for the KL term
        :return: ELBO Loss
        """
        return (self.loss_function(x, target) * self.train_size) + (kl_weight * kl)
