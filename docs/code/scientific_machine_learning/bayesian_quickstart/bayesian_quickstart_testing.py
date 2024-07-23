"""
Bayesian Quickstart Testing
===========================

This intended to be run after `bayesian_quickstart_training.py`

- Load a trained model from the file `bayesian_model.pt`
- Make deterministic predictions
- Make probabilistic predictions
- Plot probabilistic predictions
"""
# %% md
# This is the first half of a Bayesian version of the classification problem from this
# [Pytorch Quickstart tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
# This script assumes you have already run the Bayesian Quickstart Testing script and saved the optimized model
# to a file named ``bayesian_model.pth``.
#
# First, we import the necessary modules and define the BayesianNeuralNetwork class so we can load the model state
# dictionary saved in ``bayesian_model.pth``.

# %%
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml

plt.style.use("ggplot")


class BayesianNeuralNetwork(nn.Module):
    """UQpy: We rename the class and replace torch's nn.Linear with UQpy's sml.BayesianLinear"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.bayesian_linear_relu_stack = nn.Sequential(
            sml.BayesianLinear(28 * 28, 512),
            nn.ReLU(),
            sml.BayesianLinear(512, 512),
            nn.ReLU(),
            sml.BayesianLinear(512, 512),
            nn.ReLU(),
            sml.BayesianLinear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.bayesian_linear_relu_stack(x)
        return logits

# %% md
# Since the model is already trained, we only need to load test data and can ignore the training data.
# Then we use Pytorch's framework for loading a model from a state dictionary.

# Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=False,
#     transform=ToTensor(),
# )

# %%
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

device = "cpu"
network = BayesianNeuralNetwork().to(device)
model = sml.FeedForwardNeuralNetwork(network).to(device)
model.load_state_dict(torch.load("bayesian_model.pt"))

# %% md
# Here we make the deterministic prediction. We set the sample mode to ``False`` and evaluate our model on the first
# image in ``test_data``.


# %%

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
model.eval()
model.sample(False)  # UQpy: Set sample mode to False
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print("Deterministic Prediction -----")
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    print("logits:", {c: f"{float(i):.2e}" for c, i in zip(classes, pred[0])})
    print(
        "softmax(logits):",
        {c: f"{float(i):.2e}" for c, i in zip(classes, torch.softmax(pred, 1)[0])},
    )

# Make a bunch of probabilistic predictions
model.sample()
n_samples = 1_000
predictions = {c: 0 for c in classes}
with torch.no_grad():
    for i in range(n_samples):
        pred = model(x)
        pred_class = classes[pred[0].argmax(0)]
        predictions[pred_class] += 1 / n_samples
names, counts = zip(*predictions.items())
i = int(torch.argmax(torch.tensor(counts)))

print("Probabilistic Predictions -----")
print("Predicted:", classes[i], "Actual:", actual)
print("Probabilities:", {c: round(predictions[c], 3) for c in predictions})

# Plot probabilistic predictions
fig, ax = plt.subplots()
colors = ["tab:gray"] * len(classes)
colors[i] = "tab:red"
b = ax.bar(names, counts, label=counts, color=colors)
ax.bar_label(b)
ax.set_title("Bayesian NN Predictions", loc="left")
ax.set(xlabel="Classes", ylabel=f"% of {n_samples:,} Predictions")
ax.set_xticklabels(names, rotation=45)
fig.tight_layout()
plt.show()
