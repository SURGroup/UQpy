"""Training a Fourier Neural Operator with Hamiltorch's HMC

Adapting the Hamiltorch notebook found at https://github.com/AdamCobb/hamiltorch/blob/master/notebooks/hamiltorch_Bayesian_NN_example.ipynb
"""

import torch
import hamiltorch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning as sml


class FourierNeuralOperator(nn.Module):

    def __init__(self):
        """Construct a Bayesian FNO

        The lifting layers a single, deterministic, and fully connected linear layer.
        There are four Bayesian Fourier layers performing a 1d Fourier
        transform with ReLU activation functions and batch normalization in between.
        The projection layers are also a single, deterministic, and fully connected linear layer.
        """
        super().__init__()
        modes = 16
        width = 8
        self.lifting_layers = nn.Sequential(
            sml.Permutation((0, 2, 1)),
            nn.Linear(1, width),
            sml.Permutation((0, 2, 1)),
        )
        self.fourier_blocks = nn.Sequential(
            sml.Fourier1d(width, modes),
            nn.ReLU(),
            sml.Fourier1d(width, modes),
            nn.ReLU(),
            sml.Fourier1d(width, modes),
            nn.ReLU(),
            sml.Fourier1d(width, modes),
        )
        self.projection_layers = nn.Sequential(
            sml.Permutation((0, 2, 1)),
            nn.Linear(width, 1),
            sml.Permutation((0, 2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computational call of the FNO.

        Apply the lifting layers, then theFourier blocks, and finally projection layers.

        :param x: Tensor of shape :math:`(N, C, L)`
        :return: Tensor of shape :math:`(N, C, L)`
        """
        x = self.lifting_layers(x)
        x = self.fourier_blocks(x)
        x = self.projection_layers(x)
        return x


model = FourierNeuralOperator()

x_train = torch.load("initial_conditions_train.pt")
y_train = torch.load("burgers_solutions_train.pt")

# Set hyperparameters for network
tau_list = []
tau = 1.0  # /100. # iris 1/10
for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list)
params_init = hamiltorch.util.flatten(model).clone()

params_hmc = hamiltorch.sample_model(
    model,
    x_train,
    y_train,
    params_init=params_init,
    num_samples=100,
    step_size=1e-3,
    num_steps_per_sample=10,
    tau_list=tau_list,
    model_loss=nn.MSELoss(reduction="sum"),
)
