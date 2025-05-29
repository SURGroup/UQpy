import torch
from typing import Union
from beartype import beartype
from UQpy.scientific_machine_learning.baseclass import Loss


@beartype
class LpLoss(Loss):

    def __init__(
        self,
        ord: Union[int, float, str] = 2,
        dim: Union[int, tuple, None] = None,
        reduction: str = "mean",
    ):
        r"""Construct a loss function :math:`L^p(x, y)` where :math:`p=\text{dim}`

        :param ord: Order of the norm. Default: 2
        :param dim: Dimensions over which to compute the norm specified as an integer or tuple.
         If ``dim=None``, the vector is flattened before the norm is computed. Default: None
        :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
         'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
         Default: 'sum'

        .. note::
            This is an implementation of :py:class:`torch.linalg.vector_norm` as a :py:class:`torch.nn.Module`.
            This class implements most, but not all, of the :code:`vector_norm` keywords.
            See the
            `PyTorch vector_norm documentation <https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html#torch.linalg.vector_norm>`__
            for details.

        Formula
        -------

        +-------------+----------------------------------------------+
        | Ord         | Norm                                         |
        +=============+==============================================+
        | 2 (default) | :math:`\sqrt{(x-y)^2}`                       |
        +-------------+----------------------------------------------+
        | int, float  | :math:`((x-y)^n)^{1/n}`                      |
        +-------------+----------------------------------------------+
        | 0           | sum(x != 0), the number of non-zero elements |
        +-------------+----------------------------------------------+
        | -inf        | :math:`\min{|x-y|}`                          |
        +-------------+----------------------------------------------+
        | inf         | :math:`\max{|x-y|}`                          |
        +-------------+----------------------------------------------+

        where inf refers to :code:`float('inf')`, :py:class:`torch.inf`, or any equivalent object.


        Example:

        >>> loss = sml.LpLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()

        """
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss :math:`L_p(x, y)`.

        The valid shapes for ``x`` and ``y`` depend on
        `PyTorch broadcast semantics <https://pytorch.org/docs/stable/notes/broadcasting.html>`__ .

        :param x: Tensor of any shape. Must be broadcastable with ``y``
        :param y: Tensor of any shape. Must be broadcastable with ``x``.
        :return: Tensor of shape ``x`` or ``y`` (depending on broadcasting semantics).
        """
        norm = torch.linalg.vector_norm(x - y, ord=self.ord, dim=self.dim)
        if self.reduction == "none":
            return norm
        elif self.reduction == "mean":
            return torch.mean(norm)
        elif self.reduction == "sum":
            return torch.sum(norm)
        else:
            raise ValueError(
                f"UQpy: Invalid reduction={self.reduction}. Must be one of 'none', 'mean', or 'sum'"
            )
