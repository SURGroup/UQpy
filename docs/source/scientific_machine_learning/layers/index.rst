Neural Network Layers
---------------------

In this module, a layer in a neural network refers to an operation on tensor :math:`x` that maps it to a tensor :math:`y`.
Many layers have learnable parameters implemented via :class:`torch.nn.Parameter`.
They may be deterministic or have probabilistic behavior, such as the :class:`Dropout` layers.

The layers available here are designed for compatability with torch's layers, and recreate their naming and syntax
conventions as much as practical.

Notation
^^^^^^^^

We use notation consistent with PyTorch to denote the input and output tensor shapes in any given layer.
These variables are often combined to describe the shape of a tensor with many dimensions.
Unless otherwise stated, the variables are:

.. list-table:: Notation for tensor shapes
   :widths: 25 25
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`N`
     - Batch size
   * - :math:`C, C_\text{in}, C_\text{out}`
     - Number of channels in a signal
   * - :math:`L`
     - Length of a signal. Typically used for 1d signals.
   * - :math:`H, W`
     - Height, width of a signal. Typically used for 2d signals.
   * - :math:`D, H, W`
     - Depth, height, width of a signal. Typically used for 3d signals.

Layer Baseclass
^^^^^^^^^^^^^^^

The :class:`Layer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`, just as all torch layers are.
All of the Dropout layers share their own base class, as do the Bayesian layers.

This is the parent class to all layers.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom layers.
All layers use the :py:meth:`forward` method to define the forward model call.

Some documentation within the :class:`Layer` class may be inherited from PyTorch docstrings.

Methods
~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.baseclass.Layer
    :members: forward, extra_repr

------

List of Layers
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

    Bayesian Base Class <bayesian_baseclass>
    Bayesian Layers <bayesian_layers>
    Dropout Base Class <dropout_baseclass>
    Dropout Layers <dropout_layers>
    Fourier Layers <fourier_layers>
    Normalizer Layers <normalizers>
