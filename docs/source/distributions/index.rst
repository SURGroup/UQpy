Distributions
=============


This module contains functionality for all probability distributions supported in :py:mod:`UQpy`.

The :py:mod:`UQpy.distributions` module is  used  to  define  probability  distribution  objects.   These  objects  possess various
methods  that  allow the user  to:  compute  the  probability  density/mass  function ``pdf/pmf``, the cumulative
distribution  function ``cdf``, the logarithm of the pdf/pmf ``log_pdf/log_pmf``, return the moments ``moments``, draw
independent samples ``rvs`` and compute the maximum likelihood estimate of the parameters from data ``mle``.

The module contains the following parent classes - probability distributions are defined via sub-classing those parent
classes:

- :class:`.Distribution`: Parent class to all distributions.
- :class:`.Distribution1D`: Parent class to all univariate distributions.
- :class:`.DistributionContinuous1D`: Parent class to 1-dimensional continuous probability distributions.
- :class:`.DistributionDiscrete1D`: Parent class to 1-dimensional discrete probability distributions.
- :class:`.DistributionND`: Parent class to multivariate probability distributions.
- :class:`.Copula`: Parent class to copula to model dependency between marginals.



Note that the various classes of the :py:mod:`UQpy.distributions` module are written to be consistent with distributions in the
:py:mod:`scipy.stats` package :cite:`Scipy_paper`, to the extent possible while maintaining an extensible, object oriented architecture that is
convenient for operating with the other :py:mod:`UQpy` modules. All existing distributions and their methods in :py:mod:`UQpy` are
restructured from the :py:mod:`scipy.stats` package.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Distributions

    Parent Distribution Class <distribution_parent>
    Distributions Continuous 1D <distributions_continuous_1d>
    Distributions Discrete 1D <distributions_discrete_1d>
    Copulas <copulas>
    Multivariate Distributions <distributions_multivariate>
    User Defined Distributions <user_defined_distributions>

