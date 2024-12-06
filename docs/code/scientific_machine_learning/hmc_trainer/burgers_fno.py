r"""
Learning the Burgers' Operator with HMC
=======================================

In this example, we train a Fourier Neural Operator (FNO) using the implementation of Hamiltonian Monte Carlo (HMC)
provided by the Hamiltorch (https://adamcobb.github.io/journal/hamiltorch.html) library.
Our FNO learns the mapping :math:`y(x, 0) \mapsto y(x, 0.5)` where :math:`y(x,t)` is the solution to the
Burgers' equation given by

.. math:: \frac{\partial}{\partial t}u(x, t) + u(x, t) \frac{\partial}{\partial x} u(x, t) = \nu \frac{\partial^2}{\partial x^2} u(x,t)

This example is adapted from this Hamiltorch example (https://github.com/AdamCobb/hamiltorch/blob/master/notebooks/hamiltorch_Bayesian_NN_example.ipynb)
"""

# %% md
# We learn the mapping from the initial condition :math:`y(x, 0)` to a point at time :math:`t=0.5`.
# The training data for this example was computed separately and is available on GitHub.
#
# First, import the necessary modules.

# %%

import torch
import hamiltorch
import matplotlib.pyplot as plt
import torch.nn as nn
import UQpy.scientific_machine_learning as sml

# %% md
# Below we define the architecture of our neural operator as subclasses of PyTorch objects.

# %%


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


# %% md
# The training and testing data was created by solving the Burgers' equation using numerical integration.
# The dataset used here are available on our GitHub.

# %%

x_train = torch.load("initial_conditions_train.pt")
y_train = torch.load("burgers_solutions_train.pt")

# %% md
# Finally, we pass our model and the data off the Hamiltorch for the training.
# Note that Hamiltorch does not modify the parameters of the ``model`` object.
# Rather, it provides a list of parameters that have been sampled from the posterior distribution.

# %%

# Set hyperparameters for network
tau_list = []
tau = 100.0
for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list)
step_size = 1e-4
num_steps_per_sample = 100
params_init = hamiltorch.util.flatten(model).clone()

# generate the HMC samples
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
# optional, save the HMC  parameters
# torch.save(params_hmc, "params_hmc.pt")

# %% md
# That's the Fourier neural operator trained with HMC!
# The rest of this notebook visualizes the predictions from the sampled parameters.

# %%

# optional, load the HMC parameters. `params_hmc.pt` is available on our GitHub
params_hmc = torch.load("params_hmc.pt")

# Evaluate the HMC samples on testing data
x_test = torch.load("initial_conditions_test.pt")
y_test = torch.load("burgers_solutions_test.pt")
predictions, log_prob_list = hamiltorch.predict_model(
    model,
    x=x_test[0:1],
    y=y_test[0:1],
    samples=params_hmc[:],
    model_loss=nn.MSELoss(reduction="sum"),
    tau_list=tau_list,
)

# plot the results
predictions = predictions.squeeze()
mean_prediction = torch.mean(predictions, dim=0)
spacial_x = torch.linspace(0, 1, 256)
fig, ax = plt.subplots()
ax.plot(spacial_x, y_test[0].squeeze(), color="tab:blue")
ax.plot(spacial_x, mean_prediction, color="black")
ax.plot(spacial_x, predictions.T, color="gray", alpha=0.1, zorder=1)
ax.legend(["y", "Mean Prediction", "Ensemble Predictions"])
ax.set_title("HMC Predictions")
ax.set(xlabel="x", ylabel="y(x, 0.5)")

plt.show()
