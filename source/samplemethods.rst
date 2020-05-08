.. _samplemethods:

		  		   
SampleMethods
=============

This module contains functionality for all the sampling methods supported in ``UQpy``. 
The module currently contains the following classes:

- ``MCS``: Class to perform Monte Carlo sampling using ``UQpy``.
- ``LHS``: Class to perform Latin hypercube sampling using ``UQpy``.



MCS
----

``MCS``  class can be used to generate  random  draws  from  specified probability distribution(s).  The ``MCS``
class utilizes the ``Distributions`` class to define probability distributions.  The advantage of using the ``MCS``
class for ``UQpy`` operations, as opposed to simply generating samples with the ``scipy.stats`` package, is that it
allows building an object  containing  the  samples,  their  distributions  and variable names for integration with
other ``UQpy`` modules.

``MCS``  class can be imported in a python script using the following command:

>>> from UQpy.SampleMethods import MCS

For example,  to run MCS  for two independent normally distribution random variables `N(1,1)` and `N(0,1)`

>>> from UQpy.Distributions import Normal
>>> dist1 = Normal(loc=1., scale=1.)
>>> dist2 = Normal(loc=0., scale=1.)
>>> x1 = MCS(dist_object=[dist1, dist2], nsamples=5, random_state = [1,3], verbose=True)
>>> print(x1.samples)
    UQpy: Running Monte Carlo Sampling...
    UQpy: Monte Carlo Sampling Complete.
	[array([[1.62434536],
	[1.78862847]]), array([[-0.61175641],
	[ 0.43650985]]), array([[-0.52817175],
	[ 0.09649747]]), array([[-1.07296862],
	[-1.8634927 ]]), array([[ 0.86540763],
	[-0.2773882 ]])]

The ``MCS`` class can be used to run MCS for multivariate distributions

>>> from UQpy.Distributions import MVNormal
>>> dist = MVNormal(mean=[1., 2.], cov=[[4., -0.9], [-0.9, 1.]])
>>> x2 = MCS(dist_object=[dist], nsamples=5, random_state=123)
>>> print(x2.samples)
    [array([[3.38736185, 2.23541269]]), array([[0.08946208, 0.8979547 ]]), array([[2.53138343, 	3.06057229]]), array([[5.72159837, 0.30657467]]), array([[-1.71534735,  1.97285583]])]

Or for a combination of distributions

>>> from UQpy.Distributions import MVNormal, Normal
>>> dist1 = Normal(loc=1., scale=1.)
>>> dist = MVNormal(mean=[1., 2.], cov=[[4., -0.9], [-0.9, 1.]])
>>> x3 = MCS(dist_object=[dist1, dist], nsamples=5, random_state=[123, None])
>>> print(x3.samples)
	[[array([-1.0856306]) array([2.71076423, 0.1045249 ])]
 	[array([0.99734545]) array([-0.83581846,  2.72645573])]
 	[array([0.2829785]) array([2.00888019, 1.19372116])]
 	[array([-1.50629471]) array([-2.92088426,  3.20510339])]
 	[array([-0.57860025]) array([2.68866374, 3.02856845])]]
	 
In this case the number of  samples will be

>>> print(len(x3.samples.shape))
    5
and the dimension of the problem is

>>> print(len(x3.samples[0].shape))
    2

.. autoclass:: UQpy.SampleMethods.MCS
	:members:

.. toctree::
    :maxdepth: 2



	
	