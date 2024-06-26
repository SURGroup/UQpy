Losses
------

Loss Baseclass
^^^^^^^^^^^^^^

The :py:class:`Loss` is an abstract baseclass and a subclass of :py:class:`torch.nn.Module`.

This is an abstract baseclass and the parent class to all loss functions.
Like all abstract baseclasses, this cannot be instantiated but can be subclassed to write custom losses.

Methods
~~~~~~~
.. autoclass:: UQpy.scientific_machine_learning.baseclass.Loss
    :members: forward

----

List of Losses
^^^^^^^^^^^^^^

Evidence Lower Bound
~~~~~~~~~~~~~~~~~~~~

This is a placeholder for the documentation on ELBO.

.. autoclass:: UQpy.scientific_machine_learning.losses.EvidenceLowerBound
    :members: forward

------

Gaussian Kullback-Leibler
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a placeholder for KL Divergence.

.. autoclass:: UQpy.scientific_machine_learning.losses.GaussianKullbackLeiblerDivergence
    :members: forward

------

Physics Informed Loss
~~~~~~~~~~~~~~~~~~~~~

This is a placeholder for the documentation on Physics Informed Loss

.. autoclass:: UQpy.scientific_machine_learning.losses.PhysicsInformedLoss
    :members: forward

Attributes
~~~~~~~~~~
.. autoattribute:: UQpy.scientific_machine_learning.losses.PhysicsInformedLoss.adaptive_weight_data
.. autoattribute:: UQpy.scientific_machine_learning.losses.PhysicsInformedLoss.adaptive_weight_physics
