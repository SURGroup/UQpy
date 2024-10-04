if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import UQpy.scientific_machine_learning as sml
    n_filters = [1, 64, 128]
    kernel_size = 3
    out_channels = 3
    unet = Unet(n_filters, kernel_size, out_channels)
    x = torch.rand(1, 1, 512, 512)
    print(unet)
    y = unet(x)
    print()
    print(x.shape)
    print(y.shape)

    # Output the shapes
    print(f"Input shape: {x.shape}")         # (N, in_channels, H, W)
    print(f"Prediction shape: {y.shape}")  # (N, out_channels, H, W)
    print(unet)