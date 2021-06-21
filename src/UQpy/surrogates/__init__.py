"""
This module contains functionality for all the surrogate methods supported in UQpy.

The module currently contains the following classes:

- ``stochastic_reduced_order_models``: Class to estimate a discrete approximation for a continuous random variable using Stochastic
    Reduced Order Model.

- ``kriging``: Class to generate an approximate surrogate model using kriging.

- ``polynomial_chaos``: Class to generate an approximate surrogate model using Polynomial chaos.

"""

from UQpy.surrogates.polynomial_chaos import *
from UQpy.surrogates.stochastic_reduced_order_models import *
from UQpy.surrogates.kriging import *

from . import (polynomial_chaos, stochastic_reduced_order_models, kriging)
