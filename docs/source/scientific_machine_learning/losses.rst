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

.. autoclass:: UQpy.scientific_machine_learning.losses.EvidenceLowerBound
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
