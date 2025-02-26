Bayesian Layer Baseclass
------------------------

This is the parent class to all Bayesian layers.
The :class:`NormalBayesianLayer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`.

The documentation in the :py:meth:`forward` and :py:meth:`extra_repr` on this page may be inherited from PyTorch docstrings.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.NormalBayesianLayer
    :members: reset_parameters, get_bayesian_weights, sample, forward, extra_repr

Attributes
~~~~~~~~~~
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NormalBayesianLayer.parameter_shapes
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NormalBayesianLayer.sampling
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NormalBayesianLayer.prior_mu
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NormalBayesianLayer.prior_sigma
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NormalBayesianLayer.posterior_mu_initial
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.NormalBayesianLayer.posterior_rho_initial
