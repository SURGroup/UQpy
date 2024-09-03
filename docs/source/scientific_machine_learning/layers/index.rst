Neural Network Layers
---------------------

In this module, a layer in a neural network refers to an operation on tensor :math:`x` that maps it to a tensor :math:`y`.
Many layers have learnable parameters implemented via :class:`torch.nn.Parameter`.
They may be deterministic or have probabilistic behavior, such as the :class:`Dropout` layers.

The layers available here are designed for compatability with torch's layers, and recreate their naming and syntax
conventions as much as practical.

Layer Baseclass
^^^^^^^^^^^^^^^

The :class:`Layer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`,
just as all torch layers are. All of the Dropout layers share their own base class, as do the Bayesian layers.

This is the parent class to all layers.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom layers.
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

    Bayesian Base Class <bayesian_baseclass>
    Bayesian Layers <bayesian_layers>
    Dropout Base Class <dropout_baseclass>
    Dropout Layers <dropout_layers>
    Fourier Layers <fourier_layers>
    Normalizer Layers <normalizers>
