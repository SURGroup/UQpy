"""
Uncertainty Quantification with Python
========================================
"""

import pkg_resources

import UQpy.Inference
import UQpy.RunModel
import UQpy.StochasticProcess
import UQpy.Transformations
import UQpy.Surrogates
import UQpy.Utilities
import UQpy.DimensionReduction
import UQpy.Reliability
import UQpy.Distributions
import UQpy.SampleMethods

from UQpy.Inference import *
from UQpy.RunModel import *
from UQpy.StochasticProcess import *
from UQpy.Transformations import *
from UQpy.Surrogates import *
from UQpy.Utilities import *
from UQpy.DimensionReduction import *
from UQpy.Reliability import *
from UQpy.Distributions import *
from UQpy.SampleMethods import *

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None