Activation Functions
--------------------

In this module, activation functions are element-wise functions on the components of a tensor.
They define some function :math:`f(x) = y`. Activation functions typically *do not* have weights or biases.
They may be deterministic or have probabilistic behavior, such as the :class:`Dropout` layers.

Activation Function Baseclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ActivationFunction` is an abstract baseclass and a subclass of :class:`torch.nn.Module`,
just as all torch activation functions are.

This is the parent class to all activation classes.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom activation functions.
All activation functions use the :py:meth:`forward` method to define the forward model call.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.ActivationFunction
    :members: forward, extra_repr

------

List of Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

    Dropout Parent Class <dropout_parent>
    Dropout Layers <dropout_layers>