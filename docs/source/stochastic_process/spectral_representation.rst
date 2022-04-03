Spectral Representation Method
---------------------------------

The Spectral Representation Method (SRM) expands the stochastic process in a Fourier-type expansion of cosines. The
version of the SRM implemented in :py:mod:`UQpy` uses a summation of cosines with random phase angles as:

.. math:: A(t) = \sqrt{2}\sum_{i=1}^N\sqrt{2S(\omega_i)\Delta\omega}\cos(\omega_i t+\phi_i)

where :math:`S(\omega_i)` is the discretized power spectrum at frequency :math:`\omega_i`, :math:`\Delta\omega` is the
frequency discretization, and :math:`\phi_i` are random phase angles uniformly distributed in :math:`[0, 2\pi]`. For
computational efficiency, the SRM is implemented using the Fast Fourier Transform (FFT).

SpectralRepresentation Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.SpectralRepresentation` class is imported using the following command:

>>> from UQpy.stochastic_process.SpectralRepresentation import SpectralRepresentation

Methods
"""""""
.. autoclass:: UQpy.stochastic_process.SpectralRepresentation
    :members: run

Attributes
""""""""""
.. autoattribute:: UQpy.stochastic_process.SpectralRepresentation.samples
.. autoattribute:: UQpy.stochastic_process.SpectralRepresentation.n_dimensions
.. autoattribute:: UQpy.stochastic_process.SpectralRepresentation.phi
.. autoattribute:: UQpy.stochastic_process.SpectralRepresentation.n_variables


Examples
""""""""""

.. toctree::

   Spectral Representation Examples <../auto_examples/stochastic_processes/spectral/index>