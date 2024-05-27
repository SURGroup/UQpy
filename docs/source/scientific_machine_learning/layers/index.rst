Neural Network Layers
---------------------

This is a placeholder for documentation on neural network layers.

Layer Baseclass
^^^^^^^^^^^^^^^

The :class:`Layer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`,
just as all :py:mod:`torch` loss functions are.

This is the parent class to all losses.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom losses.
All loss functions use the :py:meth:`forward` method to define the forward model call.

Methods
~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.baseclass.Layer
    :members: forward

------

List of Layers
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

    Bayesian Parent Class <bayesian_parent>
    Bayesian Layers <bayesian_layers>
    Spectral Convolution <spectral_conv_1d>