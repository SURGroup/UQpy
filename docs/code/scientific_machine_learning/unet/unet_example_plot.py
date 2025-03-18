"""
Plotting the U-net example
==========================

"""
# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import axes_grid1
from torch.utils.data import DataLoader, Dataset
import UQpy.scientific_machine_learning as sml
from sklearn.metrics import mean_absolute_error

#%% Check if GPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

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

# Define dataset with appropriate shape
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


#%% Evaluate and plot
def evaluate_and_plot_predictions(unet_model, test_data, device, num_samples=5, fig_dir='figures'):
    """
    Function to load model weights, make predictions on test data, and plot results with MAE heatmap.
    """
    # Load saved weights
    weights_path = os.path.join(fig_dir, 'unet_weights.pth')
    unet_model.load_state_dict(torch.load(weights_path, map_location=device))
    unet_model.eval()
    print("Model weights loaded")

    # Prepare DataLoader for test data
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Arrays to store predictions, ground truth, inputs, and MAE scores
    X_inputs, Y_trues, Y_preds, mae_scores, mae_maps = [], [], [], [], []

    # Generate predictions and calculate MAE
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_pred = unet_model(X_batch).cpu().numpy()
            Y_preds.append(Y_pred)
            Y_trues.append(Y_batch.numpy())
            X_inputs.append(X_batch.cpu().numpy())

            # Calculate MAE per pixel
            mae_map_batch = np.abs(Y_batch.numpy() - Y_pred)
            mae_maps.append(mae_map_batch)

            # Calculate MAE per sample in the batch
            mae_batch = [mean_absolute_error(y_true.flatten(), y_pred.flatten())
                         for y_true, y_pred in zip(Y_batch.numpy(), Y_pred)]
            mae_scores.extend(mae_batch)

    # Concatenate results for consistent shapes
    X_inputs = np.concatenate(X_inputs, axis=0)
    Y_trues = np.concatenate(Y_trues, axis=0)
    Y_preds = np.concatenate(Y_preds, axis=0)
    mae_maps = np.concatenate(mae_maps, axis=0)

    # Plotting predictions with MAE in titles
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, num_samples * 5))
    fig.suptitle("Test Set Predictions with MAE", fontsize=16)
    for i in range(num_samples):
        # Input microstructure
        axes[i, 0].imshow(X_inputs[i, 0], cmap='viridis')
        axes[i, 0].set_title(f"Input Microstructure {i + 1}")
        axes[i, 0].axis('off')

        # Ground truth stress
        im = axes[i, 1].imshow(Y_trues[i, 0, :, :], cmap='viridis')
        add_colorbar(im)
        axes[i, 1].set_title("Ground Truth Stress")
        axes[i, 1].axis('off')

        # Predicted stress
        pred_mask = Y_preds[i, 0, :, :]
        im = axes[i, 2].imshow(pred_mask, cmap='viridis')
        add_colorbar(im)
        axes[i, 2].set_title(f"Predicted Stress (MAE: {mae_scores[i]:.4f})")
        axes[i, 2].axis('off')

        # MAE heatmap
        mae_map = mae_maps[i, 0, :, :]
        im = axes[i, 3].imshow(mae_map, cmap='hot')
        add_colorbar(im)
        axes[i, 3].set_title("MAE")
        axes[i, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'test_set_predictions_with_mae.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=100)
    plt.show()

# %% Define U-Net model
n_filters = [1, 16, 32, 64, 128]
kernel_size = 3
out_channels = 1
unet = sml.Unet(n_filters, kernel_size, out_channels).to(device)
X_ts = np.load('./data/X_ts.npy')
Y_ts = np.load('./data/Y_ts.npy')
test_dataset = FiberDataset(X_ts, Y_ts)

# %% Run evaluation and plotting
evaluate_and_plot_predictions(
    unet, test_dataset, device, num_samples=5, fig_dir='figures')

# %%
