.. _stochastic_process:

StochasticProcess
=================

.. automodule:: UQpy.Stochastic Process

The StochasticProcess module consists of classes and functions to generate samples of Stochastic Processes from Power
Spectrum, Bispectrums and Autocorrelation Functions. The generated Stochastic Processes can be transformed to other
marginal distributions.

SRM
----

.. autoclass:: UQpy.StochasticProcess.SRM

``SRM`` is a class for generating Stochastic Processes by Spectral Representation2442Method from a prescribed Power
Spectral Density Function.

BSRM
----

.. autoclass:: UQpy.StochasticProcess.BSRM

``BSRM`` is a class for generating Stochastic Processes by BiSpectral Representa-2492tion Method from a prescribed Power
Spectral Density Function and a Bispectral Density Function.

KLE
----

.. autoclass:: UQpy.StochasticProcess.KLE

``KLE`` is a class for generating Stochastic Processes by Karhunen Louve Expan-2541sion from a prescribed
Autocorrelation Function.

Translation
----

.. autoclass:: UQpy.StochasticProcess.Translation

``Translation`` is a class for translating Gaussian Stochastic Processes to Non-Gaussian Stochastic Processes. This
class returns the non-Gaussian samples along with the distorted Autocorrelated Function.

InverseTranslation
----

.. autoclass:: UQpy.StochasticProcess.InverseTranslation

``InverseTranslation`` is a class for translating Non-Gaussian Stochastic Processes back to Standard Gaussian Stochastic
Processes. This class returns the non-Gaussian samples along with the distorted Autocorrelated Function.
