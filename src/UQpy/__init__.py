"""
Uncertainty Quantification with Python
========================================
"""

import pkg_resources

import UQpy.distributions
import UQpy.sample_methods
import UQpy.dimension_reduction
import UQpy.inference
import UQpy.reliability
import UQpy.RunModel
import UQpy.sample_methods
import UQpy.stochastic_process
import UQpy.surrogates
import UQpy.sensitivity
import UQpy.transformations
import UQpy.Utilities


from UQpy.dimension_reduction import *
from UQpy.distributions import *
from UQpy.inference import *
from UQpy.reliability import *
from UQpy.RunModel import *
from UQpy.sample_methods import *
from UQpy.stochastic_process import *
from UQpy.surrogates import *
from UQpy.transformations import *
from UQpy.Utilities import *
from UQpy.sensitivity import *

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None