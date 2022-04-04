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
import UQpy.run_model.RunModel
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
from UQpy.run_model import *
from UQpy.sampling import *
from UQpy.stochastic_process import *
from UQpy.surrogates import *
from UQpy.transformations import *
from UQpy.utilities.Utilities import *
from UQpy.sensitivity import *
from UQpy.utilities.UQpyLoggingFormatter import *
import logging
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from warnings import filterwarnings
filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

formatter = UQpyLoggingFormatter()

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

logging.logThreads = 0
logging.logProcesses = 0

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("UQpy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
