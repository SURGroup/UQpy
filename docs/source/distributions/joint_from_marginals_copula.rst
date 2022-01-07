Joint from marginals and copula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define a joint distribution from a list of marginals and a copula to introduce dependency. :class:`.JointCopula` is a
child class of :class:`.DistributionND`.

A :class:`.JointCopula` distribution may possess a :py:meth:`cdf`, :py:meth:`pdf` and :py:meth:`log_pdf` methods if the copula allows for it
(i.e., if the copula possesses the necessary :meth:`evaluate_cdf` and :meth:`evaluate_pdf` methods - See :class:`.Copula`).

The parameters of the distribution are only stored as attributes of the marginals/copula objects. However, the
:meth:`get_parameters` and :meth:`update_parameters` methods can still be used for the joint. Note that each parameter of
the joint is assigned a unique string identifier as :code:`key_index` - where :code:`key` is the parameter name and :code:`index` the
index of the marginal (e.g., location parameter of the 2nd marginal is identified as :code:`loc_1`); and :code:`key_c` for
copula parameters.

.. autoclass:: UQpy.distributions.collection.JointCopula
    :members: