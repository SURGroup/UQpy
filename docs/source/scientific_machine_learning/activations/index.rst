Activations
-----------

In this module, activations are element-wise functions on the components of a tensor.
They define some function :math:`f(x) = y`. Activations typically *do not* have weights or biases.
They may be deterministic or have probabilistic behavior, such as the :class:`Dropout` layers.

Activation Baseclass
^^^^^^^^^^^^^^^^^^^^

The :class:`Activation` is an abstract baseclass and a subclass of :class:`torch.nn.Module`,
just as all torch activation functions are.

This is the parent class to all activation classes.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom activations.
All activations use the :py:meth:`forward` method to define the forward model call.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.Activation
    :members: forward, extra_repr

------

List of Activations
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

    Dropout Parent Class <dropout_parent>
    Dropout Layers <dropout_layers>