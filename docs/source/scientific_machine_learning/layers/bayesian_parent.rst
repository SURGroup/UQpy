Bayesian Layer Baseclass
------------------------

This is the parent class to all Bayesian layers.
The :class:`BayesianLayer` is an abstract baseclass and a subclass of :class:`torch.nn.Module`.

Methods
~~~~~~~

.. autoclass:: UQpy.scientific_machine_learning.baseclass.BayesianLayer
    :members: sample, get_weight_bias, sample_parameters, forward, extra_repr

Attributes
~~~~~~~~~~

.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.prior_mu
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.prior_sigma
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.posterior_mu_initial
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.posterior_rho_initial
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.weight_mu
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.weight_sigma
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.bias
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.bias_mu
.. autoattribute:: UQpy.scientific_machine_learning.baseclass.BayesianLayer.bias_sigma
