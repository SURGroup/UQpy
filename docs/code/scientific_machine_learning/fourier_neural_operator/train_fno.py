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


if __name__ == "__main__":
    # define datasets
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

    model = FourierNeuralOperator()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    trainer = sml.Trainer(model, optimizer, scheduler=scheduler)
    trainer.run(train_data=train_data, test_data=test_data, epochs=100)
    torch.save(model.state_dict(), "fno_state_dict.pt")

    # plot results
    fig, ax = plt.subplots()
    ax.semilogy(trainer.history["train_loss"], label="Train Loss")
    ax.semilogy(trainer.history["test_loss"], label="Test NLL")
    ax.set_title("Deterministic FNO Training History")
    ax.set(xlabel="Epoch", ylabel="MSE")
    ax.legend()
    fig.tight_layout()

    plt.show()
