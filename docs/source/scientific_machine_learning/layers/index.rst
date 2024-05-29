Neural Network Layers
---------------------

In this module, a layer in a neural network refers to an operation on tensor :math:`x` that maps it to a tensor :math:`y`.
Layers have weights and/or biases, implemented via :class:`torch.nn.Parameter`, unlike activation functions which do not.
The layers available here are designed for compatability with torch's layers, and recreate their naming and syntax
conventions as much as practical.

Layer Baseclass
^^^^^^^^^^^^^^^

The :class:`Layer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`,
just as all torch layers are.

This is the parent class to all layers.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom losses.
All layers use the :py:meth:`forward` method to define the forward model call.

Methods
~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.baseclass.Layer
    :members: forward, extra_repr

------

List of Layers
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

    Bayesian Parent Class <bayesian_parent>
    Bayesian Layers <bayesian_layers>
    Spectral Convolution <spectral_conv_1d>