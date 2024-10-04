U-net Convolutional Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.Unet` class provides the implementation of the U-net neural network (U-Net) originally introduced by Ronneberger et al. The network is originally designed for image segmentation tasks but can also be generalized to perform image-to-image regression. 

The architecture comprises a series of encoding blocks and decoding blocks. Each encoding block consists of a repeated set of a convolutional layers of kernel, a batch normalization layer followed by a nonlinear activation function and a downsampling layer together with the rectified linear unit (ReLU) activation function. The maximum pooling (max-pooling) operation is used in the downsampling layer.

The decoding blocks have the same structure as their encoding counterparts with the ex- ception that the downsampling layers are replaced by upsampling layers.The last convolutional layer of kernel size combines the features of the last multi-channel output to a single prediction. The network also includes a number of skip connections between the contracting and expanding paths, aimed at combining high resolution features with abstract feature representations from the encoding path.

The :class:`.UNeuralNetwork` class is imported using the following command:

>>> from UQpy.scientific_machine_learning import UNeuralNetwork

Methods
-------

.. autoclass:: UQpy.scientific_machine_learning.neural_networks.UNeuralNetwork
    :members: forward, optional_step_en, optional_step_dec

A schematic of the U-Net architecture is shown below:
.. figure:: ./figures/Unet_schematic.pdf
   :align: center
   :class: with-border
   :width: 600
   :alt: Diagram showing the architecture of a U-Net neural network.
   The architecture of the U-Net neural network.

**Tensor Shapes Throughout the Network:**

- **Input**: Tensor of shape :math:`(N, C_{\text{in}}, H, W)`, where:
  
  - :math:`N` is the batch size.
  - :math:`C_{\text{in}}` is the number of input channels (e.g., 1 for grayscale images, 3 for RGB images).
  - :math:`H` and :math:`W` are the height and width of the input images.

**Note on Convolutional Layer Parameters:**

Each convolutional layer accepts an input volume with width :math:`W_1`, height :math:`H_1`, and depth :math:`D_1`.

- **Input Dimensions**: :math:`W_1 \times H_1 \times D_1`
- **Number of Filters (K)**: Determines the number of filters (or kernels) used, affecting the depth :math:`D_2` of the output volume.
- **Spatial Extent (F)**: The size of each filter, typically a square (e.g., 3x3).
- **Stride (S)**: The step size with which the filters are moved across the input volume.
- **Zero Padding (P)**: Number of pixels added to the border of the input volume, enabling control over the spatial dimensions of the output volume.

- **Output Dimensions**: :math:`W_2 \times H_2 \times D_2`

  - **Output Width**: :math:`W_2 = \left( \frac{W_1 - F + 2P}{S} \right) + 1`
  - **Output Height**: :math:`H_2 = \left( \frac{H_1 - F + 2P}{S} \right) + 1`
  - **Output Depth**: :math:`D_2 = K`, corresponding to the number of filters.

Attributes
----------

.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.UNeuralNetwork.n_filters
   :annotation:

.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.UNeuralNetwork.kernel_size
   :annotation:

.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.UNeuralNetwork.out_channels
   :annotation:

.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.UNeuralNetwork.layer_type
   :annotation:

.. autoattribute:: UQpy.scientific_machine_learning.neural_networks.UNeuralNetwork.final_conv
   :annotation:

**Example Code:**
   .. code-block:: python
      :linenos:

      import torch
      import torch.nn as nn
      import UQpy.scientific_machine_learning as sml
      import sml.neural_networks.Unet as Unet
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

**References:**

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *arXiv preprint arXiv:1505.04597*.
