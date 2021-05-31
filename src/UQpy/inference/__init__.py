# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This module contains classes and functions for statistical inference from data.

The module currently contains the
following classes:

* ``InferenceModel``: Define a probabilistic model for inference.
* ``MLEstimation``: Compute maximum likelihood parameter estimate.
* ``InfoModelSelection``: Perform model selection using information theoretic criteria.
* ``BayesParameterEstimation``: Perform Bayesian parameter estimation (estimate posterior density) via markov_chain or IS.
* ``BayesModelSelection``: Estimate model posterior probabilities.
"""

from UQpy.inference.BayesModelSelection import BayesModelSelection
from UQpy.inference.InferenceModel import InferenceModel
from UQpy.inference.InformationModelSelection import InformationModelSelection
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.inference.MaximumLikelihoodEstimation import MaximumLikelihoodEstimation

from UQpy.inference.BayesModelSelection import *
from UQpy.inference.InferenceModel import *
from UQpy.inference.InformationModelSelection import *
from UQpy.inference.BayesParameterEstimation import *
from UQpy.inference.MaximumLikelihoodEstimation import *

