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
- ``markov_chain``: Class to perform Markov Chain Monte Carlo sampling.
- ``IS``: Class to perform Importance sampling.
- ``AKMCS``: Class to perform adaptive kriging Monte Carlo sampling.
- ``stratifications``: Class to perform stratified sampling.
- ``refined_stratified``: Class to perform refined stratified sampling.
- ``strata``: Class to perform stratification of the unit hypercube.
- ``Simplex``: Class to uniformly sample from a simplex.

"""

from UQpy.sample_methods.MonteCarloSampling import MCS
from UQpy.sample_methods.LatinHypercubeSampling import LHS
from UQpy.sample_methods.ImportanceSampling import IS
from UQpy.sample_methods.AdaptiveKrigingMonteCarlo import AKMCS
from UQpy.sample_methods.SimplexSampling import Simplex

from UQpy.sample_methods.refined_stratified import *
from UQpy.sample_methods.stratifications import *
from UQpy.sample_methods.markov_chain import *
from UQpy.sample_methods.strata import *

from . import (
    MarkovChainMonteCarlo, refined_stratified, stratifications, Strata
)
