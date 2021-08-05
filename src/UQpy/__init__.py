"""
Uncertainty Quantification with Python
========================================
"""

import pkg_resources

import UQpy.distributions
import UQpy.sampling
import UQpy.dimension_reduction
import UQpy.inference
import UQpy.reliability
import UQpy.RunModel
import UQpy.sampling
import UQpy.stochastic_process
import UQpy.surrogates
import UQpy.sensitivity
import UQpy.transformations
import UQpy.utilities.Utilities


from UQpy.dimension_reduction import *
from UQpy.distributions import *
from UQpy.inference import *
from UQpy.reliability import *
from UQpy.RunModel import *
from UQpy.sampling import *
from UQpy.stochastic_process import *
from UQpy.surrogates import *
from UQpy.transformations import *
from UQpy.utilities.Utilities import *
from UQpy.sensitivity import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] - %(asctime)s - File: %(filename)s - Method: %(funcName)s - %(message)s',
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None