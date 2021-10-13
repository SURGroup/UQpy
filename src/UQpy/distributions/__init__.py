"""
This module contains functionality for all probability distributions supported in ``UQpy``.

The ``distributions`` module is  used  to  define  probability  distribution  objects.   These  objects  possess various
methods  that  allow the user  to:  compute  the  probability  density/mass  function ``pdf/pmf``, the cumulative
distribution  function ``cdf``, the logarithm of the pdf/pmf ``log_pdf/log_pmf``, return the moments ``moments``, draw
independent samples ``rvs`` and compute the maximum likelihood estimate of the parameters from data ``mle``.

The module contains the following parent classes - probability distributions are defined via sub-classing those parent
classes:

- ``Distribution``: Parent class to all distributions.
- ``Distribution1D``: Parent class to all univariate distributions.
- ``DistributionContinuous1D``: Parent class to 1-dimensional continuous probability distributions.
- ``DistributionDiscrete1D``: Parent class to 1-dimensional discrete probability distributions.
- ``DistributionND``: Parent class to multivariate probability distributions.
- ``Copula``: Parent class to copula to model dependency between marginals.

"""

# pylint: disable=wildcard-import

from UQpy.distributions.baseclass import *
from UQpy.distributions.copulas import *
from UQpy.distributions.collection import *

from . import baseclass, copulas, collection
