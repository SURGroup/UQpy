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
This module contains functionality for all the stochastic process methods supported in UQpy.

The module currently contains the following classes:

- ``SRM``: Class for simulation of Gaussian stochastic processes and random fields using the Spectral Representation
  Method.
- ``BSRM``: Class for simulation of third-order non-Gaussian stochastic processes and random fields using the
  Bispectral Representation Method.
- ``KLE``: Class for simulation of stochastic processes using the Karhunen-Loeve Expansion.
- ``Translation``: Class for transforming a Gaussian stochastic process to a non-Gaussian stochastic process with
  prescribed marginal probability distribution.
- ``InverseTranslation``: Call for identifying an underlying Gaussian stochastic process for a non-Gaussian process with
  prescribed marginal probability distribution and autocorrelation function / power spectrum.

"""

from UQpy.StochasticProcess.SRM import SRM
from UQpy.StochasticProcess.BSRM import BSRM
from UQpy.StochasticProcess.KLE import KLE
from UQpy.StochasticProcess.Translation import Translation
from UQpy.StochasticProcess.InverseTranslation import InverseTranslation

from .supportive import *

from . import (
    supportive
)



