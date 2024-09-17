Losses
------

Most lost functions behave similarly to `PyTorch loss functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
The take in an input tensor :math:`x` and a target :math:`y` and return a tensor representing the distance between the two.

In contrast, the divergence functions presented here are not like the Torch loss functions.
Divergences compute a distance between the prior and posterior distributions of a Bayesian neural network.
They take a single :py:class:`torch.nn.Module` as an input to compute a distance between the prior and posterior.

Loss Baseclass
^^^^^^^^^^^^^^

The :py:class:`Loss` is an abstract baseclass and a subclass of :py:class:`torch.nn.Module`.
This is an abstract baseclass and the parent class to all loss functions.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom losses.

The documentation in the :py:meth:`forward` on this baseclass may be inherited from PyTorch docstrings.

Methods
~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.baseclass.Loss
    :members: forward

----

List of Losses
^^^^^^^^^^^^^^

:math:`L_p` Loss
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.losses.LpLoss
    :members: forward

------

Gaussian Kullback-Leibler
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.losses.GaussianKullbackLeiblerDivergence
    :members: forward

------

Monte Carlo Kullback-Leiber
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.losses.MCKullbackLeiblerDivergence
    :members: forward


------

Generalized Jenson Shannon
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.losses.GeneralizedJensenShannonDivergence
    :members: forward

------

Geometric Jenson Shannon
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.losses.GeometricJensenShannonDivergence
    :members: forward
