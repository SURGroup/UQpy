List of Bayesian Layers
^^^^^^^^^^^^^^^^^^^^^^^

All Bayesian layers use their counterparts in :py:mod:`torch.nn.functional` to define their computation.
The difference is the Bayesian implementations define their weights and biases as random variables,
rather than as deterministic parameters.
The goal of these layers is not to recreate features in Pytorch, but to provide Bayesian implementations
that match Pytorch's syntax as much as reasonable.

For example, :class:`BayesianLinear` computes :math:`y=x A^T + b` just as :class:`torch.nn.Linear` does,
and uses :class:`torch.nn.functional.linear` for the computation. For convenience, the first three pararmeters
of :class:`BayesianLinear` are identical in name and purpose to :class:`Linear`,
and are ``in_features``, ``out_features``, and ``bias``.

Bayesian Linear
~~~~~~~~~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianLinear
    :members: forward

-----

Bayesian Convolution 1D
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianConv1d
    :members: forward

-----

Bayesian Convolution 2D
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianConv2d
    :members: forward

-----

Probabilistic Layer
~~~~~~~~~~~~~~~~~~~

This is an attempt to generalize a Bayesian layer to sample weights from an arbitrary distribution.

.. autoclass:: UQpy.scientific_machine_learning.layers.ProbabilisticLayer
    :members: forward
