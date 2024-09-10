import torch
import UQpy.scientific_machine_learning as sml


class TestL0Loss:
    """This is a quirk of the pytorch vector norm function that counts non-zero elements in a tensor"""

    zeros = torch.zeros((5, 10))
    ones = torch.ones((5, 10))

    def test_all_zeros(self):
        loss_function = sml.LpLoss(ord=0)
        loss = loss_function(self.zeros, self.zeros)
        assert loss == 0

    def test_all_ones(self):
        loss_function = sml.LpLoss(ord=0)
        loss = loss_function(self.ones, self.zeros)
        assert loss == 50


class TestL1Loss:
    x = torch.tensor([[0, 1]], dtype=torch.float)
    y = torch.tensor([[1, 0]], dtype=torch.float)

    def test_loss_zero(self):
        loss_function = sml.LpLoss(ord=1)
        loss = loss_function(self.x, self.x)
        assert loss == 0.0

    def test_default(self):
        """With the default reduction and dim, compute the manhattan distance between these vectors as 2"""
        loss_function = sml.LpLoss(ord=1)
        loss = loss_function(self.x, self.y)
        assert loss == 2.0

    def test_dim(self):
        """With dim=0 and reduction=mean, compute the average manhattan distance between the vectors as 1.
        This takes an average of the manhattan distance in each component.
        """
        loss_function = sml.LpLoss(ord=1, dim=0)
        loss = loss_function(self.x, self.y)
        assert loss == 1.0


class TestL2Loss:
    x = torch.tensor([[0, 1]], dtype=torch.float)
    y = torch.tensor([[1, 0]], dtype=torch.float)

    def test_loss_zero(self):
        loss_function = sml.LpLoss()
        loss = loss_function(self.x, self.x)
        assert loss == 0.0

    def test_default(self):
        """With the defaults, compute the distance between these vectors as square root 2"""
        loss_function = sml.LpLoss()
        loss = loss_function(self.x, self.y)
        assert loss == torch.sqrt(torch.tensor(2.0))

    def test_dim(self):
        """With dim=0 and reduction=mean, compute the average distance between the vectors as 1.
        This takes an average of the distance in each component.
        """
        loss_function = sml.LpLoss(dim=0)
        loss = loss_function(self.x, self.y)
        assert loss == 1.0

    def test_dim_reduction(self):
        """With dim=0 and reduction=sum, compute the total distance between the vectors as 1
        This takes a sum of the distance in each component"""
        loss_function = sml.LpLoss(dim=0, reduction="sum")
        loss = loss_function(self.x, self.y)
        assert loss == 2.0


class TestLPositiveInfLoss:
    x = torch.tensor([[1, 2, 3]], dtype=torch.float)
    y = torch.tensor([[2, 4, 6]], dtype=torch.float)

    def test_default(self):
        loss_function = sml.LpLoss(ord=torch.inf)
        loss = loss_function(self.x, self.y)
        assert loss == 3

    def test_dim_0_reduction_mean(self):
        loss_function = sml.LpLoss(ord=torch.inf, dim=0)
        loss = loss_function(self.x, self.y)
        assert loss == 2

    def test_dim_0_reduction_none(self):
        loss_function = sml.LpLoss(ord=torch.inf, dim=0, reduction="none")
        loss = loss_function(self.x, self.y)
        assert torch.all(loss == torch.tensor([1, 2, 3]))


class TestLMinusInfLoss:
    x = torch.tensor([[1, 2, 3]], dtype=torch.float)
    y = torch.tensor([[2, 4, 6]], dtype=torch.float)

    def test_default(self):
        loss_function = sml.LpLoss(ord=-torch.inf)
        loss = loss_function(self.x, self.y)
        assert loss == 1

    def test_dim_0_reduction_mean(self):
        loss_function = sml.LpLoss(ord=-torch.inf, dim=0)
        loss = loss_function(self.x, self.y)
        assert loss == 2

    def test_dim_0_reduction_none(self):
        loss_function = sml.LpLoss(ord=-torch.inf, dim=0, reduction="none")
        loss = loss_function(self.x, self.y)
        assert torch.all(loss == torch.tensor([1, 2, 3]))
