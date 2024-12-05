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


# generate the HMC samples

# Set hyperparameters for network
tau_list = []

tau = 100.0
for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list)

step_size = 1e-4
num_steps_per_sample = int(torch.pi / (2 * step_size * tau))

params_init = hamiltorch.util.flatten(model).clone()
params_hmc = hamiltorch.sample_model(
    model,
    x_train,
    y_train,
    params_init=params_init,
    num_samples=100,
    step_size=step_size,
    num_steps_per_sample=num_steps_per_sample,
    tau_list=tau_list,
    model_loss=nn.MSELoss(reduction="sum"),
)
torch.save(params_hmc, "params_hmc.pt")

# Evaluate the HMC samples on testing data
x_test = torch.load("initial_conditions_test.pt")
y_test = torch.load("burgers_solutions_test.pt")
params_hmc = torch.load("params_hmc.pt")
predictions, log_prob_list = hamiltorch.predict_model(
    model,
    x=x_test[0:1],
    y=y_test[0:1],
    samples=params_hmc[:],
    model_loss=nn.MSELoss(reduction="sum"),
    tau_list=tau_list,
)

predictions = predictions.squeeze()
mean_prediction = torch.mean(predictions, dim=0)
spacial_x = torch.linspace(0, 1, 256)
fig, ax = plt.subplots()
ax.plot(spacial_x, y_test[0].squeeze(), color="tab:blue")
ax.plot(spacial_x, mean_prediction, color="black")
ax.plot(spacial_x, predictions.T, color="gray", alpha=0.3)
ax.legend(["y", "Mean Prediction", "Ensemble Predictions"])
ax.set_title("HMC Predictions")
ax.set(xlabel="x", ylabel="y(x, 0.5)")

plt.show()
