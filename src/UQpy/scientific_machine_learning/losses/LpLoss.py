import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.scientific_machine_learning.baseclass import Loss


@beartype
class LpLoss(Loss):
    def __init__(
        self,
        d: PositiveInteger = 2,
        p: PositiveInteger = 2,
        reduction: Annotated[str, Is[lambda s: s in ("mean", "sum", "none")]] = "mean",
    ):
        """Compute the :math:`L_p(x, y)` loss on two tensors

        :param d: ToDo: this isn't used in relative loss?
        :param p: Exponent used in the loss function
        :param reduction: Optional, defines method to simplify loss output
        """
        self.d = d
        self.p = p
        self.reduction = reduction

    def absolute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Absolute L1 Loss (Mean Absolute Error)  ToDo: this isn't used anywhere?

        :param x:
        :param y:
        :return: all_norms
        """
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)  # Assume uniform mesh
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction == "mean":
            return torch.mean(all_norms)
        elif self.reduction == "sum":
            return torch.sum(all_norms)
        else:
            return all_norms

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computation call of relative :math:`L_p` loss

        :param x: first input tensor
        :param y: second input tensor
        :return: Loss between ``x`` and ``y`` defined by :math:`L_p(x, y)`
        """
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction == "mean":
            return torch.mean(diff_norms / y_norms)
        elif self.reduction == "sum":
            return torch.sum(diff_norms / y_norms)
        else:
            return diff_norms / y_norms
