#UQpy is distributed under the MIT license.

#Copyright (C) 2018  -- Michael D. Shields

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This module contains functionality for all probability distributions supported in ``UQpy``.

The ``Distributions`` module is  used  to  define  probability  distribution  objects.   These  objects  possess various
methods  that  allow the user  to:  compute  the  probability  density/mass  function ``pdf/pmf``, the cumulative
distribution  function ``cdf``, the logarithm of the pdf/pmf ``log_pdf/log_pmf``, return the moments ``moments``, draw
independent samples ``rvs`` and compute the maximum likelihood estimate of the parameters from data ``mle``.

The module contains the following parent classes - probability distributions are defined via sub-classing those parent
classes:

- ``Distribution``: Parent class to all distributions.
- ``DistributionContinuous1D``: Parent class to 1-dimensional continuous probability distributions.
- ``DistributionDiscrete1D``: Parent class to 1-dimensional discrete probability distributions.
- ``DistributionND``: Parent class to multivariate probability distributions.
- ``Copula``: Parent class to copula to model dependency between marginals.

"""

# pylint: disable=wildcard-import

from UQpy.Distributions.baseclass import *
from UQpy.Distributions.copulas import *
from UQpy.Distributions.collection import *

from . import (
    baseclass, copulas, collection
)