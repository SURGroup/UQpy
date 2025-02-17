List of Bayesian Layers
^^^^^^^^^^^^^^^^^^^^^^^

All Bayesian layers use their counterparts in :py:mod:`torch.nn.functional` and/or
:py:mod:`UQpy.scientific_machine_learning.functional` to define their computation.
The difference between a PyTorch layer and it's Bayesian counterpart is in the defition and training of the learnable parameters.
A PyTorch layer, like :py:class:`torch.nn.Conv1d` version defines weights and biases as deterministic tensors
and learns a value for those parameters.
In contrast, UQpy's Bayesian version, like :py:class:`UQpy.scientific_machine_learning.BayesianConv1d`,
defines the weights and biases as random variables, and learns their distributions.
The purpose of these layers is not to recreate features in Pytorch, but to provide Bayesian implementations
that match Pytorch's syntax as much as possible.

For example, :class:`BayesianLinear` computes :math:`y=x A^T + b` just as :class:`torch.nn.Linear` does,
and uses :class:`torch.nn.functional.linear` for the computation. For convenience, the first three parameters
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

Bayesian Convolution 3D
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianConv3d
    :members: forward

-----


Bayesian Fourier 1D
~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianFourier1d
    :members: forward

-----

Bayesian Fourier 2D
~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianFourier2d
    :members: forward

-----

Bayesian Fourier 3D
~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.layers.BayesianFourier3d
    :members: forward
