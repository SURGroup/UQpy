"""
This module contains functionality for all the stochastic process methods supported in UQpy.

The module currently contains the following classes:

- ``SpectralRepresentation``: Class for simulation of Gaussian stochastic processes and random fields using the
    Spectral Representation Method.
- ``BispectralRepresentation``: Class for simulation of third-order non-Gaussian stochastic processes and random fields
    using the Bispectral Representation Method.
- ``KLE``: Class for simulation of stochastic processes using the Karhunen-Loeve Expansion.
- ``Translation``: Class for transforming a Gaussian stochastic process to a non-Gaussian stochastic process with
    prescribed marginal probability distribution.
- ``InverseTranslation``: Call for identifying an underlying Gaussian stochastic process for a non-Gaussian process with
    prescribed marginal probability distribution and autocorrelation function / power spectrum.

"""

from UQpy.stochastic_process.BispectralRepresentation import BispectralRepresentation
from UQpy.stochastic_process.InverseTranslation import InverseTranslation
from UQpy.stochastic_process.KarhunenLoeveExpansion import KarhunenLoeveExpansion
from UQpy.stochastic_process.SpectralRepresentation import SpectralRepresentation
from UQpy.stochastic_process.Translation import Translation

from .supportive import *

from . import supportive
