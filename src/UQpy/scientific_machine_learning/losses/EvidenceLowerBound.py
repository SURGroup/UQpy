import torch
from beartype import beartype
from UQpy.scientific_machine_learning.baseclass.Loss import Loss
from UQpy.utilities.ValidationTypes import PositiveInteger, PositiveFloat


@beartype
class EvidenceLowerBound(Loss):
    def __init__(
        self,
        train_size: PositiveInteger,
        loss_function: torch.nn.Module = torch.nn.NLLLoss(),
        **kwargs,
    ):
        """Construct an Evidence Lower Bound (ELBO) function

        :param train_size: Number of training samples
        :param loss_function: Function to compute loss on prior and posterior. Default: ``torch.nn.NLLLoss``
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
        """Forward computational call

        :param x: Predicted values
        :param target: Target values
        :param kl: Kullback-Leibler Divergence between variational posterior and prior
        :param kl_weight: Weighting factor for the KL term
        :return: ELBO Loss
        """
        return (self.loss_function(x, target) * self.train_size) + (kl_weight * kl)

    def extra_repr(self) -> str:
        return f"train_size={self.train_size}, loss_function={self.loss_function}"
