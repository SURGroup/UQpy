"""
Uncertainty Quantification with Python
========================================
"""

import pkg_resources

import UQpy.DimensionReduction
import UQpy.Distributions
import UQpy.Inference
import UQpy.Reliability
import UQpy.RunModel
import UQpy.SampleMethods
import UQpy.StochasticProcess
import UQpy.Surrogates
import UQpy.Transformations
import UQpy.Utilities
from UQpy.DimensionReduction import *
from UQpy.Distributions import *
from UQpy.Inference import *
from UQpy.Reliability import *
from UQpy.RunModel import *
from UQpy.SampleMethods import *
from UQpy.StochasticProcess import *
from UQpy.Surrogates import *
from UQpy.Transformations import *
from UQpy.Utilities import *

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None