Fourier Neural Operator (FNO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of the Fourier Neural Operator as defined by Li 2021.
There is no explict Fourier Neural Operator class, but rather an implementation of the Fourier layers defined by Li in
Fig. 2.

Schematically, we represent the Fourier Neural Operator as having three parts shown below.
After the input layer, we lift the number of features up to the width of the Fourier layers, apply the Fourier layers,
then project the Fourier representation to the desired number of out channels.

.. figure:: ./figures/fourier_network_diagram.pdf
   :align: center
   :class: with-border
   :width: 800
   :alt: A diagram showing the architecture of a Fourier neural network network.

   The architecture of a generic Fourier Neural Network.


The following example considers a 1-dimensional input, with 1 in channel and 6 out channels.
We build a large Fourier neural operator to map the in channel to the out channels.
We construct Fourier Neural Operator with a standard linear lifting layer from the number of
in channels to the width of the fourier layers, 4 Fourier layers, followed by two projection layers that steps from
the width down to 16 features, apply the rectified linear unit, then step down again to the desired 6 out channels.
The permutation is necessary just to correctly align the dimensions of the tensors. It does not perform any computation.

The Fourier network constructed in the example below takes an input of size :math:`(N, C_{\text{in}, L)` and outputs
a tensor of size :math:`(N, C_{\text{out}, L)` where :math:`N` is the batch size, :math:`C` denotes a number of channels,
and :math:`L` is the length of the signal.

.. code-block:: python
    :linenos:

    import torch
    import torch.nn as nn
    import UQpy.scientific_machine_learning as sml

    in_channels = 1
    out_channels = 6
    width = 32
    modes = 50

    fno = nn.Sequential(
        nn.Linear(in_channels, width),  # lifting layer
        sml.Permutation((0, 2, 1)),
        sml.Fourier1d(width, modes),  # start of Fourier layers
        nn.ReLU(),
        sml.Fourier1d(width, modes),
        nn.ReLU(),
        sml.Fourier1d(width, modes),
        nn.ReLU(),
        sml.Fourier1d(width, modes),
        sml.Permutation((0, 2, 1)),
        nn.Linear(width, 16),  # start of projection layers
        nn.ReLU(),
        nn.Linear(16, out_channels)
    )

    N = 25
    L = 100
    x = torch.rand((N, L, in_channels))
    prediction = fno(x)  # prediction has shape (N, L, out_channels)
    print(x.shape)
    print(prediction.shape)
    print(fno)