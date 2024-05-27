Activation Functions
--------------------

This is a placeholder for documentation on activation functions.

Activation Function Baseclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ActivationFunction` is an abstract baseclass and a subclass of :class:`torch.nn.Module`,
just as all :py:mod:`torch` activation functions are.

This is the parent class to all activation classes.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom losses.
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