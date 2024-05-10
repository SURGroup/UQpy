Layers
------

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

----

List of Layers
^^^^^^^^^^^^^^

Dropout Layer
~~~~~~~~~~~~~

This is a placeholder for the documentation on Dropout layers.

.. autoclass:: UQpy.scientific_machine_learning.layers.Dropout
    :members: forward

______

Bayesian Layer
~~~~~~~~~~~~~~

This is a placeholder for the documentation on Bayesian layers.

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianLayer
    :members: forward


Probabilistic Layer
~~~~~~~~~~~~~~~~~~~

This is an attempt to generalize a Bayesian layer to sample weights from an arbitrary distribution.

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticLayer
    :members: forward
