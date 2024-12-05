"""
Learning the Burgers' Operator
==============================

In this example, we train a Fourier Neural Operator to learn the mapping :math:`y(x, 0) \mapsto y(x, 0.5)`
where :math:`y(x,t)` is the solution to the Burgers' equation given by

..math:: \frac{\partial}{\partial t}u(x, t) + u(x, t) \frac{\partial}{\partial x} u(x, t) = \nu \frac{\partial^2}{\partial x^2} u(x,t)

"""

# %% md
# We learn the mapping from the initial condition :math:`y(x, 0)` to a point at time :math:`t=0.5`.
# The training data for this example was computed separately and is available on GitHub.
#
# First, import the necessary modules.

# %%

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml

torch.manual_seed(123)

plt.style.use("ggplot")
logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)


# %% md
# Below we define the data and architecture of our neural operator as subclasses of PyTorch objects.

# %%


class BurgersDataset(Dataset):

    def __init__(self, filename_initial_condition: str, filename_solution: str):
        r"""Construct a dataset for training a FNO to learn the Burgers' operator :math:`\mathcal{B}:y(x,0) \to y(x, t^*)`

        :param filename_initial_condition: Relative path to the `.pt` file containing a torch tensor of shape :math:`(N_\text{samples}, 1, N_x)`
        :param filename_solution: Relative path to the `.pt` file containing a torch tensor of shape :math:`(N_\text{samples}, 1, N_x, N_t)`
        """

        self.filename_initial_condition = filename_initial_condition
        self.filename_solution = filename_solution
        self.initial_conditions = torch.load(self.filename_initial_condition)
        self.burgers_solution = torch.load(self.filename_solution)
        self.n_samples = self.initial_conditions.size(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.initial_conditions[item], self.burgers_solution[item]


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


# %% md
# We create two instances of the :code:`BurgersDataset` class, one for training data and one for testing data.

# %%

train_dataset = BurgersDataset(
    "initial_conditions_train.pt",
    "burgers_solutions_train.pt",
)
test_dataset = BurgersDataset(
    "initial_conditions_test.pt",
    "burgers_solutions_test.pt",
)
train_data = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_data = DataLoader(test_dataset)

# %% md
# Finally, we are ready to instantiate the model and train is using the Adam optimizer
# and UQpy's :py:class:`sml.Trainer`.
# The model is quite small, with only a few thousand parameters,
# and we only train for 100 epochs to save on computational time.
# In many applications much larger Fourier neural operators are needed and are trained for longer.

# %%

model = FourierNeuralOperator()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
trainer = sml.Trainer(model, optimizer, scheduler=scheduler)
trainer.run(train_data=train_data, test_data=test_data, epochs=100)
# torch.save(model.state_dict(), "fno_state_dict.pt")  # optionally save the trained model

# %% md
# The hard part is done! The rest of this example plots the loss history and a prediction from the trained model.

# %%

# plot loss history
fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"], label="Train Loss")
ax.semilogy(trainer.history["test_loss"], label="Test NLL")
ax.set_title("Deterministic FNO Training History")
ax.set(xlabel="Epoch", ylabel="MSE")
ax.legend()
fig.tight_layout()

# plot FNO prediction

# optional, load weights from save file. fno_state_dict.pt is available on our GitHub.
# model.load_state_dict(torch.load("fno_state_dict.pt", weights_only=True))

x = torch.linspace(0.0, 1.0, 256)  # spacial domain
fig, ax = plt.subplots()
initial_condition, burgers_solution = train_dataset[0:1]
with torch.no_grad():
    prediction = model(initial_condition)

initial_condition = initial_condition.squeeze()
burgers_solution = burgers_solution.squeeze()
prediction = prediction.squeeze()
ax.plot(x, initial_condition, label="$y(x,0)$", color="black", linestyle="dashed")
ax.plot(
    x,
    burgers_solution,
    label="$y(x, 0.5)$",
    color="black",
)
ax.plot(x, prediction, label="FNO $\hat{y}(x, 0.5)$", color="tab:blue")
ax.set_title("Deterministic FNO on Training Data", loc="left")
ax.set(ylabel="$y(x, t)$", xlabel="Time $t$")
ax.legend(fontsize="small", ncols=2)

plt.show()
