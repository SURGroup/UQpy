import beartype
import torch
import torch.functional as F
from UQpy.scientific_machine_learning.baseclass.Loss import Loss


@beartype
class EvidenceLowerBound(Loss):
    def __init__(self, train_size: int, beta: float, **kwargs):  
        """

        :param train_size: Number of training samples
        :param beta: Weighting factor for the KL term
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.train_size = train_size
        self.beta = beta


    def forward(self, x, target, kl) -> torch.Tensor:
        """
        
        :param x:
        :param target:
        :param kl:
        :return:
        """
        return F.nll_loss(x, target, reduction="mean") * self.train_size + self.beta * kl
