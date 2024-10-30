# %% Imports 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import UQpy.scientific_machine_learning as sml
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import axes_grid1
# %%
# Check if GPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Selected device: {device}")

# Colorbar function
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

# Visualization of data
def plot_images(X, Y, title, num_samples):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    fig.suptitle(title, fontsize=16)
    plt.ion()
    for i in range(num_samples):
        # Adjusted for transposed shape
        axes[i, 0].imshow(X[i, 0], cmap='viridis')
        axes[i, 0].set_title(f"Input microstructure {i + 1}")
        axes[i, 0].axis('off')

        im = axes[i, 1].imshow(Y[i, 0], cmap='viridis')
        add_colorbar(im)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.ioff()

## % Define dataset with appropriate shape
class FiberDataset(Dataset):
    def __init__(self, X, Y, num_samples=None, img_size=None):
        self.X = np.transpose(X, (3, 2, 0, 1))  # Transpose to (N, C_in, H, W)
        self.Y = np.transpose(Y, (3, 2, 0, 1))  # Transpose to (N, C_out, H, W)
        # Select only one channel, keep dimensions
        self.X = self.X[:, :1, :, :]
        self.Y = self.Y[:, :1, :, :]

        if num_samples is not None:
            self.X = self.X[:num_samples]
            self.Y = self.Y[:num_samples]
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)
# %% Load data and define trainers
X_tr = np.load('./data/X_tr.npy')
Y_tr = np.load('./data/Y_tr.npy')
X_val = np.load('./data/X_val.npy')
Y_val = np.load('./data/Y_val.npy')

train_dataset = FiberDataset(X_tr, Y_tr)
val_dataset = FiberDataset(X_val, Y_val)

plot_images(train_dataset.X, train_dataset.Y, "Training Set Examples", 3)
plot_images(val_dataset.X, val_dataset.Y, "Validation Set Examples", 3)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# %% Define U-Net model
n_filters = [1, 16, 32, 64, 128]
kernel_size = 3
out_channels = 1
unet = sml.Unet(n_filters, kernel_size, out_channels).to(device)

# Set up optimizer and scheduler
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.005)

# Set up Trainer and train
trainer = sml.Trainer(unet, optimizer, scheduler=scheduler)
print("Starting Training...")
trainer.run(train_data=train_loader, test_data=val_loader,
            epochs=1000)  
print("Training Complete.")

# Save model weights and plot
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)
weights_path = os.path.join(fig_dir, 'unet_weights.pth')
torch.save(unet.state_dict(), weights_path)
print(f"Model weights saved to {weights_path}")

# Plot training history
fig, ax = plt.subplots()
ax.plot(trainer.history["train_loss"], label="Train Loss")
ax.plot(trainer.history["test_loss"], label="Validation Loss")
ax.set_title("Training and Validation Loss")
ax.set(xlabel="Epoch", ylabel="Loss")
ax.legend()
fig_path = os.path.join(fig_dir, 'training_validation_loss.png')
fig.savefig(fig_path)
print(f"Figure saved to {fig_path}")