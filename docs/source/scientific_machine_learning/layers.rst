Neural Network Layers
---------------------

Layer Baseclass
^^^^^^^^^^^

This is an abstract baseclass and the parent class to all layers.
Like all abstract baseclasses, this cannot be called instantiated but can be subclassed to write custom layers.

Methods
~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.baseclass.Layer
    :members: forward

----

List of Layers
^^^^^^^^^^^^^^

Dropout Layer
~~~~~~~

This is a placeholder for the documentation on Dropout layers.

.. autoclass:: UQpy.scientific_machine_learning.layers.Dropout
    :members: forward

______

Bayesian Layer
~~~~~~~~

This is a placeholder for the documentation on Bayesian layers.

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianLayer
    :members: forward


Probabilistic Layer
~~~~~~~~~~~~~~~~~~

This is an attempt to generalize a Bayesian layer to sample weights from an arbitrary distribution.

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticLayer
    :members: forward
