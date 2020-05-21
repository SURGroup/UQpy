.. _transformations:

Transformations
================

This module contains functionality for all the transformations methods supported in ``UQpy``. 
The module currently contains the following classes:

- ``Nataf``: Class to perform the Nataf isoprobabilistic transformation.
- ``Correlate``: Class to induce correlation sto standard normal random variables.
- ``Uncorrelate``: Class to remove correlation from standard normal random variables.


Nataf
-------

``Nataf`` is   a   class   for   transforming   random   variables   using   the   `Nataf` transformation  and  calculating  the  correlation  distortion.   
According  to  the Nataf transformation theory, a `n`-dimensional dependent random vector :math:`\textbf{x}=[X_1,...,X_n]` for which the  marginal cumulative distributions :math:`F_i(x_i)`  and the correlattion matrix :math:`\textbf{C}_x=[\xi_{ij}]` are known, can be transformed (component-wise) to standard normal random vector :math:`\textbf{z}=[Z_1,...,Z_n]` with correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]` through the transformation:

.. math:: Z_{i}=\Phi^{-1}(F_i(X_{i})) 

where :math:`\Phi(\cdot)` is the standard normal cumulative distribution function. 

This transformation causes a `correlation distortion`; the correlation coefficient between the standard normal variables :math:`Z_i` and :math:`Z_j` is not equal to its counterpart in the  random space (:math:`\rho_{ij} \neq \xi_{ij}`). In this case :math:`\rho_{ij}` can be determined through the following integral equation:

.. math:: \xi_{ij} =\dfrac{1}{\sigma_{X_i}\sigma_{X_j}}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\left(X_i-\mu_{X_i}\right)\left(X_j-\mu_{X_j}\right)\phi_2(Z_i,Z_j;\rho_{ij})dZ_idZ_j

where :math:`X_i =F_i^{-1}(\Phi(Z_{i}))` and :math:`\phi_2(\cdot)` is the bivariate standard normal probability density function. The integration is directly evaluated using a quadratic two-dimensional Gauss-Legendre integration.

After the correlation matrix :math:`\textbf{C}_z=[\rho_{ij}]` is determined, the mutually independent standard normal vector :math:`\textbf{U}=[U_1,...,U_n]` can be calculated as: 

.. math:: \textbf{u} = \mathbf{H}^{-1} \mathbf{z}

where :math:`\mathbf{H}` is the lower triangular matrix resulting from the Cholesky decomposition of the correlation matrix, i.e. :math:`\mathbf{C_z}=\mathbf{H}\mathbf{H}^\intercal`. 

The Jacobian of the Nataf transformation can be also estimated with the ``Nataf`` class as:

.. math:: \mathbf{J_{xu}}=\dfrac{\partial\mathbf{x}}{\partial\mathbf{u}}= \dfrac{\partial\mathbf{x}}{\partial\mathbf{z}} =\left[\dfrac{\phi(Z_i)}{f_i(X_i)}\right]\mathbf{H}.

The ``Nataf`` class  is the parent class for the ``Forward`` class which performs the Nataf transformation and the ``Inverse`` class which performs the inverse Nataf transformation:

.. math:: \mathbf{z}=\mathbf{H}\mathbf{u}^\intercal
.. math:: X_{i}=F_i^{-1}(\Phi(Z_{i}))

The ``Forward`` and ``Inverse``  classes can be imported in a python script using the following command:

>>> from UQpy.Transformations import Forward, Inverse


.. autoclass:: UQpy.Transformations.Nataf
    :members: 
	
Forward
~~~~~~~~~~

This is the class that runs the forward Nataf transformation. 

Usage example:

>>> from UQpy.Transformations import Forward
>>> from UQpy.Distributions import Gamma, Lognormal
>>> dist1 = Gamma(4.0, loc=0.0, scale=1.0)
>>> dist2 = Lognormal(s=2., loc=0., scale=np.exp(1))

>>> x = MCS(dist_object=[dist1,dist2], nsamples=5, random_state=[4, 5],   verbose=True)
>>> print(x.samples)
	UQpy: Running Monte Carlo Sampling.
	UQpy: Monte Carlo Sampling Complete.
	[[  3.76433963   6.56961331]
 	[  4.70973474   1.40250468]
 	[  2.07115224 351.26550921]
 	[  5.16162027   1.64183701]
 	[  2.92258993   3.38454568]]

Provide a correlation matrix between the random variables
 
>>> Rho = np.array([[1.0, 0.3], [0.3, 1.0]])
>>> y = Forward(dist_object=[dist1,dist2], samples=x.samples, cov=Rho)
>>> print('Uncorrelated standard normal samples:', y.u)
	Uncorrelated standard normal samples:
	[[ 0.04812662  0.44122749]
 	[ 0.50108916 -0.33087015]
 	[-1.01129063  2.43077119]
 	[ 0.6964691  -0.25209213]
 	[-0.42496562  0.10960984]]
>>> print(Correlated standard normal samples:', y.z)
	Correlated standard normal samples:
	[[ 0.04812662  0.26471437]
	 [ 0.50108916  0.26487295]
	 [-1.01129063  0.35727675]
	 [ 0.6964691   0.47326136]
	 [-0.42496562 -0.31113451]]
>>> print('Distorder correlation matrix:', y.Cz)
	Distorder correlation matrix:
	[[1.         0.86261522]
 	[0.86261522 1.        ]]
>>> print('Jacobian of the transformation of the first sample:', y.Jxu[0])
	Jacobian of the transformation of the first sample:
	[[0.51722095 0.        ]
 	[0.0656519  0.03850003]]

.. autoclass:: UQpy.Transformations.Forward
    :members:

Inverse
~~~~~~~~~~

This is the class that runs the inverse Nataf transformation. 

Usage example:

>>> from UQpy.Transformations import Inverse
>>> from UQpy.Distributions import Gamma, Lognormal, Normal
>>> dist = Normal(loc=0., scale=1.)
>>> dist1 = Gamma(4.0, loc=0.0, scale=1.0)
>>> dist2 = Lognormal(s=2., loc=0., scale=np.exp(1))

>>> x = MCS(dist_object=[dist, dist], nsamples=5, random_state=[2, 1], verbose=True)
>>> print(x.samples)
	UQpy: Running Monte Carlo Sampling.
	UQpy: Monte Carlo Sampling Complete.
	[[-0.41675785  1.62434536]
 	[-0.05626683 -0.61175641]
 	[-2.1361961  -0.52817175]
 	[ 1.64027081 -1.07296862]
 	[-1.79343559  0.86540763]]

Provide a correlation matrix between the random variables
 
>>> Rho = np.array([[1.0, 0.3], [0.3, 1.0]])
>>> q = Inverse(dist_object=[dist1,dist2], samples=x.samples, cov=Rho)
>>> print('Uncorrelated standard normal samples:', q.z)
	Correlated standard normal samples:
	[[-0.41675785  1.32203398]
 	[-0.05626683 -0.58319075]
 	[-2.1361961  -1.33855585]
 	[ 1.64027081 -0.32728366]
 	[-1.79343559  0.07578496]]
>>> print(Correlated standard normal samples:', y.x)
	Correlated standard normal samples:
	[[ 2.93597783 70.01368531]
	 [ 3.56611641  0.79970463]
	 [ 0.95463233  0.94521439]
	 [ 7.73948904  0.3179258 ]
	 [ 1.23023994 15.34539239]]
>>> print('Distorder correlation matrix:', y.Cz)
	Distorder correlation matrix:
	[[1.         0.12103547]
	 [0.12103547 1.        ]]
>>> print('Jacobian of the transformation of the first sample:', y.Jxu[0])
	Jacobian of the transformation of the first sample:
	[[ 0.61211596  0.        ]
	 [-0.26714931  0.00499148]]

.. autoclass:: UQpy.Transformations.Inverse
    :members:
	
Correlate
----------

This is a class to correlate standard normal variables according to 

.. math:: \mathbf{z}^\intercal = \mathbf{Hu}^\intercal

where :math:`\mathbf{u}` is the vector of uncorrelated standard normal samples and  :math:`\mathbf{H}` is the lower triangular matrix obtained from the `Cholesky` decomposition of the covariance matrix :math:`\mathbf{C_Z}`, i.e. :math:`\mathbf{C_Z}=\mathbf{H}\mathbf{H}^\intercal`.

For example, if we want to correlate standard normal samples 

>>> from UQpy.Transformations import Correlate
>>> from UQpy.Distributions import Normal
>>> dist1 = Normal(loc=0.0, scale=1.0)
>>> dist2 = Normal(loc=0.0, scale=1.0)

>>> u = MCS(dist_object=[dist1,dist2], nsamples=1000).samples

.. image:: _static/Transformations_uncorrelate.png
   :scale: 100 %
   :alt: Unorrelated standard normal samples
   :align: center
   
according to the correlation matrix 

>>> cov = np.array([[1.0, 0.9], [0.9, 1.0]])

we can do it by simply typing:
    
>>> z = Correlate(u, cov).z

.. image:: _static/Transformations_correlate.png
   :scale: 100 %
   :alt: Correlated standard normal samples obtained with the ``Correlate`` class.
   :align: center

.. autoclass:: UQpy.Transformations.Correlate
    :members:

Uncorrelate
------------

This is a class to uncorrelate standard normal variables according to 

.. math:: \mathbf{u}^\intercal = \mathbf{H^{-1}z}^\intercal

where :math:`\mathbf{z}` is the vector of correlated standard normal samples and :math:`\mathbf{H}` is the lower triangular matrix obtained from the `Cholesky` decomposition of the covariance matrix :math:`\mathbf{C_Z}`, i.e. :math:`\mathbf{C_Z}=\mathbf{H}\mathbf{H}^\intercal`.

For example, we can use the ``Uncorrelate`` class to uncorrelate the samples **z** from the ``Correlate`` class example by simply typing:
    
>>> u = Uncorrelate(z, cov).u

.. image:: _static/Transformations_uncorrelate.png
   :scale: 100 %
   :alt: Unorrelated standard normal samples obtained with the ``Uncorrelate`` class.
   :align: center

.. autoclass:: UQpy.Transformations.Uncorrelate
    :members:

.. toctree::
    :maxdepth: 2



	
	