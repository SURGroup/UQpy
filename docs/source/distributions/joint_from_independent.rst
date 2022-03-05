Joint from independent marginals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define a joint distribution from its independent marginals. :class:`.JointIndependent` is a child class of
:class:`.DistributionND`.

Such a multivariate distribution possesses the following methods, on condition that all its univariate marginals
also possess them:

* :py:meth:`pdf`, :py:meth:`log_pdf`, :py:meth:`cdf`, :py:meth:`rvs`, :py:meth:`fit`, :py:meth:`moments`.

The parameters of the distribution are only stored as attributes of the marginal objects. However, the
:py:meth:`get_parameters` and :py:meth:`update_parameters` method can still be used for the joint. Note that, for this purpose, each
parameter of the joint is assigned a unique string identifier as :code:`key_index` - where :code:`key` is the parameter name and
:code:`index` the index of the marginal (e.g., location parameter of the 2nd marginal is identified as :code:`loc_1`).

The :class:`.JointIndependent` class is imported using the following command:

>>> from UQpy.distributions.collection.JointIndependent import JointIndependent

.. autoclass:: UQpy.distributions.collection.JointIndependent
    :members: