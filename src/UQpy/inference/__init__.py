"""
This module contains classes and functions for statistical inference from data.

The module currently contains the
following classes:

* ``InferenceModel``: Define a probabilistic model for inference.
* ``MLEstimation``: Compute maximum likelihood parameter estimate.
* ``InfoModelSelection``: Perform model selection using information theoretic criteria.
* ``BayesParameterEstimation``: Perform Bayesian parameter estimation (estimate posterior density) via mcmc or IS.
* ``BayesModelSelection``: Estimate model posterior probabilities.
"""

from UQpy.inference.BayesModelSelection import BayesModelSelection
from UQpy.inference.inference_models import *
from UQpy.inference.InformationModelSelection import InformationModelSelection
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.inference.MLE import MLE

from UQpy.inference.BayesModelSelection import *
from UQpy.inference.InformationModelSelection import *
from UQpy.inference.BayesParameterEstimation import *
from UQpy.inference.MLE import *

