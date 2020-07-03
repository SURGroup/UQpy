.. _stochastic_process_doc:

StochasticProcess
=================

The ``StochasticProcess`` module consists of classes and functions to generate samples of stochastic processes from prescribed properties of the process (e.g. power spectrum, bispectrum and/or autocorrelation function). The existing classes rely on stochastic expansions taking the following general form,

.. math:: A(x) = \sum_{i=1}^N \theta(\omega) \phi(x),

such that the process can be expressed in terms of a set of uncorrelated random variables, :math:`\theta(\omega)`, and deterministic basis functions :math:`\phi(x)`.


The ``StochasticProcess`` module supports simulation of uni-variate, multi-variate, multi-dimensional, Gaussian and non-Gaussian stochastic processes. Gaussian stochasitc processes can be simulated using the widely-used Spectral Representation Method ([1]_, [2]_, [3]_, [4]_) and the Karhunen-Loeve Expansion ([5]_, [6]_, [7]_). Non-Gaussian stochastic processes can be generated through higher-order spectral representations ([8]_, [9]_, [10]_) or through a nonlinear transformation from a Gaussian stochastic process to a prescribed marginal distribution using translation process theory [11]_. Modeling of arbitrarily distributed random processes with specified correlation and/or power spectrum can be performed using the Iterative Translation Approximation Method (ITAM) ([12]_, [13]_) for inverse translation process modeling.

.. automodule:: UQpy.StochasticProcess

As with other modules of ``UQpy``, adding simulation methods requires the user to build a new class to support the desired functionality. It does not require modification of any existing classes or methods.

Spectral Representation Method
---------------------------------

The Spectral Representation Method (SRM) expands the stochastic process in a Fourier-type expansion of cosines. The version of the SRM implemented in ``UQpy`` uses a summation of cosines with random phase angles as:

.. math:: A(t) = \sqrt{2}\sum_{i=1}^N\sqrt{2S(\omega_i)\Delta\omega}\cos(\omega_i t+\phi_i)

where :math:`S(\omega_i)` is the discretized power spectrum at frequency :math:`\omega_i`, :math:`\Delta\omega` is the frequency discretization, and :math:`\phi_i` are random phase angles uniformly distributed in :math:`[0, 2\pi]`. For computational efficiency, the SRM is implemented using the Fast Fourier Transform (FFT).

SRM Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.StochasticProcess.SRM
	:members:


Third-order Spectral Representation Method
-------------------------------------------

The third-order Spectral Representation Method (or Bispectral Representation Method) is a generalization of the SRM for processes posessing a known power spectrum and bispectrum. Implementation follows from references [8]_ and [9]_. The multi-variate formulation from reference [10]_ is not currently implemented.

BSRM Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.StochasticProcess.BSRM
	:members:


Karhunen Loève Expansion
----------------------------

The Karhunen Loève Expansion expands the stochastic process as follows:

.. math:: A(x) = \sum_{i=1}^N \sqrt{\lambda_i} \theta_i(\omega)f_i(x)

where :math:`\theta_i(\omega)` are uncorrelated standardized random variables and :math:`\lambda_i` and :math:`f_i(x)` are the eigenvalues and eigenvectors repsectively of the covariance function :math:`C(x_1, x_2)`. 

KLE Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.StochasticProcess.KLE
	:members:

Non-Gaussian Translation Processes
-----------------------------------

A translation processes results from the nonlinear transformation of a Gaussian stochastic process. The standard translation process, introduced and extensively studied by Grigoriu [11]_ and implemented in ``UQpy`` arises from the pointwise transformation of a Gaussian process through the inverse cumulative distribution function of a specified marginal probability distribution as:

.. math:: X(t) = F^{-1}(\Phi(Y(t)))

where :math:`Y(x)` is a Gaussian random process with zero mean and unit standard deviation, :math:`\Phi(x)` is the standard normal cumulative distribution function and :math:`F^{-1}(\cdot)` is the inverse cumulative distribution function of the prescribed non-Gaussian probability distribution.

The nonlinear translation in the equation above has the effect of distorting the correlation of the stochastic process. That is, if the Gaussian process has correlation function :math:`\rho(\tau)` where :math:`\tau=t_2-t_1` is a time lag, then the non-Gaussian correlation function of the process :math:`X(t)` can be compuated as:

.. math:: \xi(\tau) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} F^{-1}(\Phi(y_1)) F^{-1}(\Phi(y_2)) \phi(y_1, y_2; \rho(\tau)) dy_1 dy_2

where :math:`\phi(y_1, y_2; \rho(\tau))` is the bivariate normal probability density function having correlation :math:`\rho(\tau)`. This operation is known as correlation distortion and is not, in general, invertible. That is, given the non-Gaussian correlation function :math:`\xi(\tau) ` and an arbitrarily prescribed non-Gaussian probability distribution with cdf :math:`F(x)`, it is not always possible to identify a correponding Gaussian process having correlation function :math:`\rho(\tau)` that can be translated to this non-Gaussian process through the equations above [11]_. 

This gives rise to the challenge of inverse translation process modeling, where the objective is to find the an underlying Gaussian process and its correlation function such that it maps as closely as possible to the desired non-Gaussian stochastic process with its arbitrarily prescribed distribution and correlation function. This problem is solved in ``UQpy`` using the Iterative Translation Approximation Method (ITAM) developed in [12]_ for processes described by their power spectrum (and using ``SRM`` for simulation) and in [13]_ for processes described through their correlation function (and using ``KLE`` for simulation). This is implemented in the ``InverseTranslation`` class.

Translation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.StochasticProcess.Translation
	:members:


.. autoclass:: UQpy.StochasticProcess.InverseTranslation
	:members:


.. [1] Shinozuka, M. and Jan, C. M. (1972). "Digital simulation of random processes and its applications," Journal of Sound and Vibration. 25: 111–128.

.. [2] Shinozuka, M. and Deodatis, G. (1991) "Simulation of Stochastic Processes by Spectral representation" Applied Mechanics Reviews. 44.

.. [3] Shinozuka, M. and Deodatis, G. (1996) "Simulation of multi-dimensional Gaussian stochastic fields by spectral representation," Applied Mechanics Reviews. 49: 29–53.

.. [4] Deodatis, G. "Simulation of Ergodic Multivariate Stochastic Processes," Journal of Engineering Mechanics. 122: 778–787.

.. [5] Huang, S.P., Quek, S. T., and Phoon, K. K. (2001). "Convergence study of the truncated Karhunen-Loeve expansion for simulation of stochastic processes," International Journal for Numerical Methods in Engineering. 52: 1029–1043.

.. [6] Phoon K.K., Huang S.P., Quek S.T. (2002). "Simulation of second-order processes using Karhunen–Loève expansion." Computers and Structures 80(12):1049–60.

.. [7] Betz, W., Papaioannou, I., & Straub, D. (2014). "Numerical methods for the discretization of random fields by means of the Karhunen–Loève expansion." Computer Methods in Applied Mechanics and Engineering, 271: 109-129.

.. [8] Shields, M.D. and Kim, H. (2017). "Simulation of higher-order stochastic processes by spectral representation," Probabilistic Engineering Mechanics. 47: 1-15.

.. [9] Vandanapu, L. and Shields, M.D. (2020). "3rd-order Spectral Representation Methods: Multi-dimensional random fields with fast Fourier transform implementation." arXiv:1910.06420

.. [10] Vandanapu, L. and Shields, M.D. (2020). "3rd-order Spectral Representation Methods: Ergodic multi-variate random processes with fast Fourier transform." arXiv:1910.06420

.. [11] Grigoriu, M. (1995). "Applied Non-Gaussian Processes," Pretice Hall. 

.. [12] Shields, M.D., Deodatis, G., and Bocchini, P. (2011). "A simple and efficient methodology to approximate a general non-Gaussian stationary stochastic process by a translation process," Probabilistic Engineering Mechanics. 26: 511-519.

.. [13] Kim, H. and Shields, M.D. (2105). "Modeling strongly non-Gaussian non-stationary stochastic processes using the Iterative Translation Approximation Method and Karhunen–Loève expansion," Computers and Structures. 161: 31-42.

.. toctree::
    :maxdepth: 2
