"""
Bayesian Quickstart Testing
===========================
"""

# %% md
# This is the second half of a Bayesian version of the classification problem from this Pytorch Quickstart tutorial:
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#
# This script assumes you have already run the Bayesian Quickstart Testing script and saved the optimized model
# to a file named ``bayesian_model.pt``. This script
#
# - Loads a trained model from the file ``bayesian_model.pt``
# - Makes deterministic predictions
# - Makes probabilistic predictions
# - Plots probabilistic predictions
#
# First, we import the necessary modules and define the BayesianNeuralNetwork class, so we can load the model state
# dictionary saved in ``bayesian_model.pt``.

# %%
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml

plt.style.use("ggplot")


class BayesianNeuralNetwork(nn.Module):
    """UQpy: Replace torch's nn.Linear with UQpy's sml.BayesianLinear"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            sml.BayesianLinear(28 * 28, 512),  # nn.Linear(28 * 28, 512)
            nn.ReLU(),
            sml.BayesianLinear(512, 512),  # nn.Linear(512, 512)
            nn.ReLU(),
            sml.BayesianLinear(512, 512),  # nn.Linear(512, 512)
            nn.ReLU(),
            sml.BayesianLinear(512, 10),  # nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# %% md
# Since the model is already trained, we only need to load test data and can ignore the training data.
# Then we use Pytorch's framework for loading a model from a state dictionary.

# %%

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

device = torch.device("cpu")
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
    print("----- Deterministic Prediction")
    print(f"Predicted: {predicted}, Actual: {actual}")
    print(f"{'Class'.ljust(11)} {'Logits'.rjust(7)} {'softmax(Logits)'}")
    for i, c in enumerate(classes):
        print(f"{c.ljust(12)} {pred[0, i]: 6.2f} {torch.softmax(pred, 1)[0, i]: 15.2e}")

# %% md
# Just like the torch tutorial, our model correctly identifies this image as an ankle boot.
# Next, we show how to make probabilistic predictions by turning on our model's ``sampling`` mode.
# We feed the same image of an ankle boot into our model 1,000 times and each time the weights and biases
# are sampled from the distributions that were learned during training.
# Rather than one static output, we get a distribution of 1,000 predictions!

# %%

model.sample()
n_samples = 1_000
logits = torch.empty((n_samples, len(classes)))
with torch.no_grad():
    for i in range(n_samples):
        logits[i] = model(x)

# %% md
# Those few lines are all it takes to make Bayesian predictions.
# The rest of this example converts the predicted logits into class predictions.
# The predicted distribution is visualized in the histogram below, which
# shows the majority of predictions classify the image as an ankle boot.
# Interestingly, the two other shoe classes (sneaker and sandal) are also common predictions.

# %%

predicted_classes = torch.argmax(logits, dim=1)
predicted_counts = torch.bincount(predicted_classes)
predictions = {c: int(i) for c, i in zip(classes, predicted_counts)}
i = torch.argmax(predicted_counts)

print("----- Probabilistic Predictions")
print("Most Commonly Predicted:", classes[i], "Actual:", actual)
print(f"{'Class'.ljust(11)} Probability")
for c in classes:
    print(f"{c.ljust(11)} {predictions[c] / n_samples: 11.3f}")

# Plot probabilistic class predictions
fig, ax = plt.subplots()
colors = plt.cm.tab10_r(torch.linspace(0, 1, 10))
b = ax.bar(classes, predicted_counts, label=predicted_counts, color=colors)
ax.bar_label(b)
ax.set_title("Bayesian NN Predictions", loc="left")
ax.set(xlabel="Classes", ylabel="Counts")
ax.set_xticklabels(classes, rotation=45)
fig.tight_layout()

# %% md
# In addition to visualizing the class predictions, we can look at the logit values output by our Bayesian model.
# Below we plot a histogram of the logits for Sandal, Sneaker, Bag, and Ankle Boot.
# The other classes are omitted since they were never predicted by our model.
#
# Notice that for Bag, the histogram peaks near zero.
# This aligns with the fact that Bag is a very unlikely prediction from our model.
# Contrast that histogram with the one for Ankle Boot, which is very likely to take on a high value.
# We interpret this as our model being very likely to predict Ankle Boot.

# %%

# Plot probabilistic logit predictions
softmax_logits = torch.softmax(logits, 1)
fig, ax = plt.subplots()
ax.hist(softmax_logits[:, 9], label="Ankle Boot", bins=20, facecolor=colors[9])
for i in (5, 7, 8):  # loop over Sandal, Sneaker, Bag
    ax.hist(
        softmax_logits[:, i],
        label=classes[i],
        bins=20,
        edgecolor=colors[i],
        facecolor="none",
        linewidth=2,
    )
ax.set_title("Bayesian NN softmax(logits)", loc="left")
ax.set(xlabel="softmax(logit) Value", ylabel="Log(Counts)")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()


# %% md
# We can also visualize these distributions and how they correlate with one another with a pair plot.
# The packages seaborn and pandas combine to easily make a pair plot, but need to be installed separately.

# %%

# Use pandas and seaborn for easy pair plots
# import seaborn
# import pandas as pd
#
# df = pd.DataFrame({classes[i]: softmax_logits[:, i] for i in (5, 7, 8, 9)})
# seaborn.pairplot(df, corner=True, plot_kws={"alpha": 0.2, "edgecolor": None})

plt.show()
