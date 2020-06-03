.. _samplemethods:

		  		   
SampleMethods
=============

.. automodule:: UQpy.SampleMethods


MCS
----

The ``MCS`` class generates random samples from a specified probability distribution(s).  The ``MCS`` class utilizes the ``Distributions`` class to define probability distributions.  The advantage of using the ``MCS`` class for ``UQpy`` operations, as opposed to simply generating samples with the ``scipy.stats`` package, is that it allows building an object containing the samples and their distributions for integration with other ``UQpy`` modules.

Class Descriptions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.SampleMethods.MCS
	:members:


LHS
----

The ``LHS`` class generates random samples from a specified probability distribution(s) using Latin hypercube sampling. LHS has the advantage that the samples generated are uniformly distributed over each marginal distribution. LHS is perfomed by dividing the range of each random variable into N bins with equal probability mass, where N is the required number of samples, generating one sample per bin, and then randomly pairing the samples.

Adding New Latin Hypercube Design Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	 
The ``LHS`` class offers a variety of methods for pairing the samples in a Latin hypercube design. These are specified by the `criterion` parameter (i.e. 'random', 'centered', 'minmax', 'correlate'). However, adding a new method is straightforward. This is done by creating a new method that contains the algorithm for pairing the samples. This method takes as input the randomly generated samples in equal probability bins in each dimension and returns a set of samples that is paired according to the user's desired criterion. The user may also pass criterion-specific parameters into the custom method. These parameters are input to the ``LHS`` class through the `**kwargs`. The output of this function should be a numpy array of at least two-dimensions with the first dimension being the number of samples and the second dimension being the number of variables . An example user-defined criterion is given below:

	
>>> def criterion(samples):
>>> 	lhs_samples = np.zeros_like(samples)
>>> 	for j in range(samples.shape[1]):
>>> 		order = np.random.permutation(samples.shape[0])
>>> 		lhs_samples[:, j] = samples[order, j]
>>> 	return lhs_samples


Class Descriptions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.SampleMethods.LHS
	:members:

	
MCMC
----

The goal of Markov Chain Monte Carlo is to draw samples from some probability distribution :math:`p(x)=\frac{\tilde{p}(x)}{Z}`, where :math:`\tilde{p}(x)` is known but :math:`Z` is hard to compute (this will often be the case when using Bayes' theorem for instance). In order to do this, the theory of a Markov chain, a stochastic model that describes a sequence of states in which the probability of a state depends only on the previous state, is combined with a Monte Carlo simulation method, see e.g. ([1]_, [2]_). More specifically, a Markov Chain is built and sampled from whose stationary distribution is the target distribution :math:`p(x)`.  For instance, the Metropolis-Hastings (MH) algorithm goes as follows:

* initialize with a seed sample :math:`x_{0}`
* walk the chain: for :math:`k=0,...` do:
   * sample candidate :math:`x^{\star} \sim Q(\cdot \vert x_{k})` for a given Markov transition probability :math:`Q`
   * accept candidate (set :math:`x_{k+1}=x^{\star}`) with probability :math:`\alpha(x^{\star} \vert x_{k})`, otherwise propagate last sample :math:`x_{k+1}=x_{k}`.
   
.. math:: \alpha(x^{\star} \vert x_{k}):= \min \left\{ \frac{\tilde{p}(x^{\star})}{\tilde{p}(x)}\cdot \frac{Q(x \vert x^{\star})}{Q(x^{\star} \vert x)}, 1 \right\}
     
The transition probability :math:`Q` is chosen by the user (see input `proposal` of the MH algorithm, and careful attention must be given to that choice as it plays a major role in the accuracy and efficiency of the algorithm. The following figure shows samples accepted (blue) and rejected (red) when trying to sample from a 2d Gaussian distribution using MH, for different scale parameters of the proposal distribution. If the scale is too small, the space is not well explored; if the scale is too large, many candidate samples will be rejected, yielding a very inefficient algorithm. As a rule of thumb, an acceptance rate of 10\%-50\% could be targeted (see `Diagnostics` in the `Utilities` module).

.. image:: _static/SampleMethods_MCMC_samples.png
   :scale: 40 %
   :alt: IS weighted samples
   :align: center

Finally, samples from the target distribution will be generated only when the chain has converged to its stationary distribution, after a so-called burn-in period. Thus the user would often reject the first few samples (see input `nburn`). Also, the chain yields correlated samples; thus to obtain i.i.d. samples from the target distribution, the user should keep only one out of n samples (see input `jump`). This means that the code will perform in total nburn + jump * N evaluations of the target pdf to yield N i.i.d. samples from the target distribution (for the MH algorithm with a single chain).

The parent class for all MCMC algorithms is the ``MCMC class``, which defines the inputs that are common to all MCMC algorithms, along with the ``run`` method that is being called to run the chain. Any given MCMC algorithm is a child class of MCMC that overwrites the main ``run_one_iteration`` method.

Adding New MCMC Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to add a new MCMC algorithm, a user must create a child class of ``MCMC``, and overwrite the ``run_one_iteration`` method that propagates all the chains forward one iteration. Such a new class may use any number of additional inputs compared to the ``MCMC`` base class. The reader is encouraged to have a look at the ``MH`` class and its code to better understand how a particular algorithm should fit the general framework. 

A useful note is that the user has access to a number of useful attributes / utility methods as the algorithm proceeds, such as:

* the attribute ``evaluate_log_target`` (and possibly ``evaluate_log_target_marginals`` if marginals were provided) is created at initialization. It is a callable that simply evaluates the log-pdf of the target distribution at a given point `x`. It can be called within the code of a new sampler as ``log_pdf_value = self.evaluate_log_target(x)``. 
* the `nsamples` and `nsamples_per_chain` attributes indicate the number of samples that have been stored up to the current iteration (i.e., they are updated dynamically as the algorithm proceeds),
* the `samples` attribute contains all previously stored samples. Cautionary note: `self.samples` also contains trailing zeros, for samples yet to be stored, thus to access all previously stored samples at a given iteration the user must call ``self.samples[:self.nsamples_per_chain]``, which will return an `ndarray` of size (self.nsamples_per_chain, self.nchains, self.dimension) ,
* the `log_pdf_values` attribute contains all previously stored log target values. Same cautionary note as above,
* the ``_update_acceptance_rate`` method updates the `acceptance_rate` attribute of the sampler, given a (list of) boolean(s) indicating if the candidate state(s) were accepted at a given iteration,
* the ``_check_methods_proposal`` method checks whether a given proposal is adequate (i.e., has ``rvs`` and ``log_pdf``/``pdf`` methods).


Class Descriptions
^^^^^^^^^^^^^^^^^^^^


.. autoclass:: UQpy.SampleMethods.MCMC
   :members:

MH
~~~~~

.. autoclass:: UQpy.SampleMethods.MH
	:members:

MMH
~~~~~
   
.. autoclass:: UQpy.SampleMethods.MMH
	:members:

Stretch
~~~~~~~~
   
.. autoclass:: UQpy.SampleMethods.Stretch
	:members:

DRAM
~~~~~~~
   
.. autoclass:: UQpy.SampleMethods.DRAM
	:members:

DREAM
~~~~~~~
   
.. autoclass:: UQpy.SampleMethods.DREAM
	:members:



   
IS
----

Importance sampling (IS) is based on the idea of sampling from an alternate distribution and reweighting the samples to be representative of the target distribution (perhaps concentrating sampling in certain regions of the input space that are of greater importance). This often enables efficient evaluations of expectations :math:`E_{ \textbf{x} \sim p} [ f(\textbf{x}) ]` where :math:`f( \textbf{x})` is small outside of a small region of the input space. To this end, a sample :math:`\textbf{x}` is drawn from a proposal distribution :math:`q(\textbf{x})` and re-weighted to correct for the discrepancy between the sampling distribution :math:`q` and the true distribution :math:`p`. The weight of the sample is computed as 

.. math:: w(\textbf{x}) = \frac{p(\textbf{x})}{q(\textbf{x})}

If :math:`p` is only known up to a constant, i.e., one can only evaluate :math:`\tilde{p}(\textbf{x})`, where :math:`p(\textbf{x})=\frac{\tilde{p}(\textbf{x})}{Z}`, IS can be used by further normalizing the weights (self-normalized IS). The following figure shows the weighted samples obtained when using IS to estimate a 2d Gaussian target distribution :math:`p`, sampling from a uniform proposal distribution :math:`q`.

.. image:: _static/SampleMethods_IS_samples.png
   :scale: 40 %
   :alt: IS weighted samples
   :align: center
   
   
Class Descriptions
^^^^^^^^^^^^^^^^^^^^
   
.. autoclass:: UQpy.SampleMethods.IS
   :members:
   
.. [1] Gelman et al., "Bayesian data analysis", Chapman and Hall/CRC, 2013
.. [2] R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014


.. toctree::
    :maxdepth: 2



	
	