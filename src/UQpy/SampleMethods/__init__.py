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
This module contains functionality for all the sampling methods supported in ``UQpy``.

The module currently contains the following classes:

- ``MCS``: Class to perform Monte Carlo sampling.
- ``LHS``: Class to perform Latin hypercube sampling.
- ``MCMC``: Class to perform Markov Chain Monte Carlo sampling.
- ``IS``: Class to perform Importance sampling.
- ``AKMCS``: Class to perform adaptive Kriging Monte Carlo sampling.
- ``STS``: Class to perform stratified sampling.
- ``RSS``: Class to perform refined stratified sampling.
- ``Strata``: Class to perform stratification of the unit hypercube.
- ``Simplex``: Class to uniformly sample from a simplex.

"""

from .MCS import MCS
from .LHS import LHS
from .IS import IS
from .AKMCS import AKMCS
from .Simplex import Simplex

from .RSS import *
from .STS import *
from .MCMC import *
from .Strata import *
from .baseclass import *

from . import (
    baseclass, MCMC, RSS, STS, Strata
)
