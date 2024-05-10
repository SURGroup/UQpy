import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass import Loss


class PhysicsInformedLoss(Loss):
    def __init__(
        self,
        loss_function_physics: nn.Module,
        loss_function_data: nn.Module,
        self_adaptive: bool = False,
        **kwargs
    ):
        """Compute Physics Informed Loss

        :param loss_function_physics: Physical equation defined at collocation points
        :param loss_function_data: Loss function defined on the data points
        :param self_adaptive: If ``True`` use self-adaptive weights on loss terms. If ``False``, do not use weights.
        :param kwargs: Keyword arguments passed to ``nn.Module``
        """
        super().__init__(**kwargs)
        self.self_adaptive = self_adaptive
        self.loss_function_physics = loss_function_physics
        self.loss_function_data = loss_function_data

        self.adaptive_weight_physics = 1.0
        """Parameter to control the weight of the physical loss. 
        
        If ``self_adapative`` is ``False``, this attribute is 1.
        If ``self_adapative`` is ``True``, this attribute is a ``torch.nn.Parameter``.
        """
        self.adaptive_weight_data = 1.0
        """Parameter to control the weight of the data loss. 

        If ``self_adapative`` is ``False``, this attribute is 1.
        If ``self_adapative`` is ``True``, this attribute is a ``torch.nn.Parameter``.
        """
        if self.self_adaptive:
            self.adaptive_weight_physics = torch.nn.Parameter()
            self.adaptive_weight_data = torch.nn.Parameter()

    def forward(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        collocation: torch.Tensor,
    ) -> torch.Tensor:
        """Forward model call

        :param prediction: Model prediction
        :param truth: True value that prediction is trying to achieve
        :param collocation: Inputs that satisfy physical loss
        :return: Sum of physical loss at collocation and data loss at prediction
        """
        return (
            self.adaptive_weight_physics
            * self.loss_function_physics(collocation)
        ) + (self.adaptive_weight_data * self.loss_function_data(prediction, truth))
