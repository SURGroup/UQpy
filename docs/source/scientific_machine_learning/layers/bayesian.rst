Bayesian Layer Baseclass
~~~~~~~~~~~~~~~~~~~~~~~~

This is a placeholder for the documentation on Bayesian layers.

.. autoclass:: UQpy.scientific_machine_learning.baseclass.BayesianLayer
    :members: sample, get_weight_bias

List of Bayesian Layers
^^^^^^^^^^^^^^^^^^^^^^^

Bayesian Linear
~~~~~~~~~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianLinear
    :members: forward

Bayesian Convolution 1D
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianConv1d
    :members: forward

Bayesian Convolution 2D
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianConv2d
    :members: forward


Probabilistic Layer
~~~~~~~~~~~~~~~~~~~

This is an attempt to generalize a Bayesian layer to sample weights from an arbitrary distribution.

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticLayer
    :members: forward
