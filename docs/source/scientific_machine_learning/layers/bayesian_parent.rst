Bayesian Layer Baseclass
------------------------

This is the parent class to all Bayesian layers.
The :class:`BayesianLayer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.BayesianLayer
    :members: reset_parameters, get_bayesian_weights, sample

Attributes
~~~~~~~~~~
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.parameter_shapes
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.sampling
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.prior_mu
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.prior_sigma
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.posterior_mu_initial
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.posterior_rho_initial
