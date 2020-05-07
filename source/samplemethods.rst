.. _samplemethods:

		  		   
SampleMethods
=============

This module contains functionality for all the sampling methods supported in ``UQpy``. 
The module currently contains the following classes:

- ``MCS``: Class to perform Monte Carlo sampling using ``UQpy``.
- ``LHS``: Class to perform Latin hypercube sampling using ``UQpy``.
- ``MCMC``: Class to perform sampling using Markov chains using ``UQpy``.
- ``IS``: Class to perform Importance sampling using ``UQpy``.



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
>>> x1 = MCS(dist_object=[dist1, dist2], nsamples=5, random_state = 123, verbose=True)
>>> print(x1.samples)
    UQpy: Running Monte Carlo Sampling...
    UQpy: Monte Carlo Sampling Complete.
    [[-0.0856306  -1.0856306 ]
     [ 1.99734545  0.99734545]
     [ 1.2829785   0.2829785 ]
     [-0.50629471 -1.50629471]
     [ 0.42139975 -0.57860025]]

The ``MCS`` class can be used to run MCS for multivariate distributions

>>> from UQpy.Distributions import MVNormal
>>> dist = MVNormal(mean=[1., 2.], cov=[[4., -0.9], [-0.9, 1.]])
>>> x2 = MCS(dist_object=[dist], nsamples=5, random_state=123)
>>> print(x2.samples)
    [[ 3.38736185  2.23541269]
     [ 0.08946208  0.8979547 ]
     [ 2.53138343  3.06057229]
     [ 5.72159837  0.30657467]
     [-1.71534735  1.97285583]]

Or for a combination of distributions

>>> from UQpy.Distributions import MVNormal, Normal
>>> dist1 = Normal(loc=1., scale=1.)
>>> dist = MVNormal(mean=[1., 2.], cov=[[4., -0.9], [-0.9, 1.]])
>>> x3 = MCS(dist_object=[dist1, dist], nsamples=5, random_state=123)
>>> print(x3.samples)
    [[-0.0856306   3.38736185  2.23541269]
     [ 1.99734545  0.08946208  0.8979547 ]
     [ 1.2829785   2.53138343  3.06057229]
     [-0.50629471  5.72159837  0.30657467]
     [ 0.42139975 -1.71534735  1.97285583]]
	 
In this case the size of the samples will be

>>> print(x3.samples.shape)
    (5, 3)


.. autoclass:: UQpy.SampleMethods.MCS
	:members:
	
MCMC
----

The goal of Markov Chain Monte Carlo is to draw samples from some probability distribution :math:`p(x)=\frac{\tilde{p}(x)}{Z}`, where :math:`\tilde{p}(x)` is known but $Z$ is hard to compute (this will often be the case when using Bayes' theorem for instance). In order to do this, the theory of a Markov chain, a stochastic model that describes a sequence of states in which the probability of a state depends only on the previous state, is combined with a Monte Carlo simulation method. More specifically, a Markov Chain is built and sampled from whose stationary distribution is the target distribution :math:`p(x)`.  For instance, the Metropolis-Hastings (MH) algorithm goes as follows:

* initialize with a seed sample :math:`x_{0}`
* walk the chain: for :math:`k=0,...` do:
   * sample candidate :math:`x^{\star} \sim Q(\cdot \vert x_{k})` for a given Markov transition probability :math:`Q`
   * accept candidate (set :math:`x_{k+1}=x^{\star}`) with probability :math:`\alpha(x^{\star} \vert x_{k})`, otherwise propagate last sample :math:`x_{k+1}=x_{k}`.
   
.. math:: \alpha(x^{\star} \vert x_{k}):= \min \left\{ \frac{\tilde{p}(x^{\star})}{\tilde{p}(x)}\cdot \frac{Q(x \vert x^{\star})}{Q(x^{\star} \vert x)}, 1 \right\}
     
The transition probability :math:`Q` is chosen by the user (see input `proposal` of the MH algorithm, and careful attention must be given to that choice as it plays a major role in the accuracy and efficiency of the algorithm. The following figure shows samples accepted (blue) and rejected (red) when trying to sample from a 2d Gaussian distribution using MH, for different scale parameters of the proposal distribution. If the scale is too small, the space is not well explored; if the scale is too large, many candidate samples will be rejected, yielding a very inefficient algorithm. As a rule of thumb, an acceptance rate of 10\%-50\% could be targeted (see `Diagnostics` in the `Utilities` module).

.. image:: _static/MCMC_samples.png
   :scale: 40 %
   :alt: IS weighted samples
   :align: center

Finally, samples from the target distribution will be generated only when the chain has converged to its stationary distribution, after a so-called burn-in period. Thus the user would often reject the first few samples (see input `nburn`). Also, the chain yields correlated samples; thus to obtain i.i.d. samples from the target distribution, the user should keep only one out of n samples (see input `jump`). This means that the code will perform in total nburn + jump * N evaluations of the target pdf to yield N i.i.d. samples from the target distribution (for the MH algorithm with a single chain).

The parent class for all MCMC algorithms is the ``MCMC class``, which defines the inputs that are common to all MCMC algorithms, along with the *run* method that is being called to run the chain. Any given MCMC algorithm is a sub-class of MCMC that overwrites the main *run_iterations* method.

.. autoclass:: UQpy.SampleMethods.MCMC
   :members:

MH
~~~~~

.. autoclass:: UQpy.SampleMethods.MH

MMH
~~~~~
   
.. autoclass:: UQpy.SampleMethods.MMH

Stretch
~~~~~~~
   
.. autoclass:: UQpy.SampleMethods.Stretch

DRAM
~~~~~~~
   
.. autoclass:: UQpy.SampleMethods.DRAM

DREAM
~~~~~~~
   
.. autoclass:: UQpy.SampleMethods.DREAM
   
   
IS
----

Importance sampling (IS) is based on the idea of concentrating sampling in certain regions of the input space, allowing efficient evaluations of expectations :math:`E_{ \textbf{x} \sim p} [ f(\textbf{x}) ]` where :math:`f( \textbf{x})` is small outside of a small region of the input space. To this end, a sample :math:`\textbf{x}` is drawn from a proposal distribution :math:`q(\textbf{x})` and re-weighted to correct for the discrepancy between the sampling distribution :math:`q` and the true distribution :math:`p`. The weight of the sample is computed as 

.. math:: w(\textbf{x}) = \frac{p(\textbf{x})}{q(\textbf{x})}

If :math:`p` is only known up to a constant, i.e., one can only evaluate :math:`\tilde{p}(\textbf{x})`, where :math:`p(\textbf{x})=\frac{\tilde{p}(\textbf{x})}{Z}`, IS can be used by further normalizing the weights (self-normalized IS). The following figure shows the weighted samples obtained when using IS to estimate a 2d Gaussian target distribution :math:`p`, sampling from a uniform proposal distribution :math:`q`.

.. image:: _static/IS_samples.png
   :scale: 40 %
   :alt: IS weighted samples
   :align: center
   
   
.. autoclass:: UQpy.SampleMethods.IS
   :members:


.. toctree::
    :maxdepth: 2



	
	