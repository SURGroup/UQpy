"""
Learning a Stress Field with U-Nets
===================================

"""

# %% md
# In this example we will approximate the mapping from one image to another using a U-Net.
# The architecture in this example is based on the work of Pasparakis et al. :cite:`pasparakis2024bayesian`
# The data for this example was provided by Bhaduri et al. :cite:`bhaduri2022unetdata`
# This example
#
# 1. Loads the training and validation data
# 2. Defines and trains the U-net
# 3. Visualizes the results
#
# First, import the necessary modules.

# %%

import logging
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import UQpy.scientific_machine_learning as sml

logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)

# %% md
# **Define the training and testing data**
#
# Below we define a Dataset that loads the data saved as NumPy arrays and converts them to PyTorch tensors.
# We plot the input-output pairs in the training data to get a sense for what our data looks like.

# %%


class FiberDataset(Dataset):
    def __init__(
        self,
        microstructure_filename: str = "microstructure.pt",
        mask_filename: str = "masks.pt",
    ):
        """Construct a Dataset to train a U-net

        :param microstructure_filename: File containing a torch tensor of shape :math:`(N, C_\text{in}, H, W)`
        :param mask_filename: File containing a torch tensor of shape :math:`(N, C_\text{out}, H, W)`
        """
        self.x = torch.load(microstructure_filename)
        self.y = torch.load(mask_filename)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = FiberDataset()
train_dataset, val_dataset = random_split(
    dataset, [40, 10], generator=torch.Generator().manual_seed(0)
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset)

# Plot input-output pair
fig, (ax0, ax1) = plt.subplots(ncols=2)
fig.suptitle("U-net Training Data")
ax0.imshow(dataset.x[0, 0])
ax0.set_title("Input: Microstructure")
ax1.imshow(dataset.y[0, 0])
ax1.set_title("Output: Mask")
fig.tight_layout()

# %% md
# **Train the U-net**
#
# With our data ready, we can define and train the U-net!
# UQpy's U-net class assumes that we are working with 2D image data, so we just need to pass a list of filter sizes
# to the class, along with our desired kernel size and number of output channels.

# %%

n_filters = [1, 16, 32, 64, 128]
kernel_size = 3
out_channels = 1
unet = sml.Unet(n_filters, kernel_size, out_channels)

optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
trainer = sml.Trainer(unet, optimizer, scheduler=scheduler)
trainer.run(train_data=train_loader, test_data=val_loader, epochs=50)

# %% md
# **Visualize the results**
#
# Finally, we plot the loss history and the predictions made by the optimized network.

# %%

fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"], label="Train Loss")
ax.semilogy(trainer.history["test_loss"], label="Validation Loss")
ax.set_title("U-Net Training History")
ax.set(xlabel="Epoch", ylabel="MSE Loss")
ax.legend()
fig.tight_layout()

i = train_dataset.indices[0]
x = dataset.x[i:i+1]
y = dataset.y[i:i+1].squeeze()
with torch.no_grad():
    prediction = unet(x).squeeze()
error = prediction - y
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
ax0.imshow(prediction)
ax1.imshow(y)
ax2.imshow((prediction - y).squeeze(), cmap="bwr", norm=colors.TwoSlopeNorm(vcenter=0))
ax0.set_title("Prediction")
ax1.set_title("Truth")
ax2.set_title("Error")
fig.tight_layout()

plt.show()
