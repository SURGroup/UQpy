Joint from independent marginals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define a joint distribution from its independent marginals. :class:`.JointIndependent` is a child class of
:class:`.DistributionND`.

Such a multivariate distribution possesses the following methods, on condition that all its univariate marginals
also possess them:

* ``pdf``, ``log_pdf``, ``cdf``, ``rvs``, ``fit``, ``moments``.

The parameters of the distribution are only stored as attributes of the marginal objects. However, the
*get_parameters* and *update_parameters* method can still be used for the joint. Note that, for this purpose, each
parameter of the joint is assigned a unique string identifier as `key_index` - where `key` is the parameter name and
`index` the index of the marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

.. autoclass:: UQpy.distributions.collection.JointIndependent
    :members: