Third-order Spectral Representation Method
-------------------------------------------

The third-order Spectral Representation Method (or Bispectral Representation Method) is a generalization of the
SpectralRepresentation for processes possessing a known power spectrum and bispectrum. Implementation follows from
references :cite:`StochasticProcess8` and :cite:`StochasticProcess9`. The multi-variate formulation from reference
:cite:`StochasticProcess10` is not currently implemented.

BispectralRepresentation Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.BispectralRepresentation` class is imported using the following command:

>>> from UQpy.stochastic_process.BispectralRepresentation import BispectralRepresentation

Methods
"""""""
.. autoclass:: UQpy.stochastic_process.BispectralRepresentation
    :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.n_dimensions
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.bispectrum_amplitude
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.bispectrum_real
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.bispectrum_imaginary
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.biphase
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.phi
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.samples
.. autoattribute:: UQpy.stochastic_process.BispectralRepresentation.n_variables


Examples
""""""""""

.. toctree::

   Bispectral representation Examples <../auto_examples/stochastic_processes/bispectral/index>
