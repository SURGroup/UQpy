MCMC
----

The goal of Markov Chain Monte Carlo is to draw samples from some probability distribution
:math:`p(x)=\frac{\tilde{p}(x)}{Z}`, where :math:`\tilde{p}(x)` is known but :math:`Z` is hard to compute (this will
often be the case when using Bayes' theorem for instance). In order to do this, the theory of a Markov chain, a
stochastic model that describes a sequence of states in which the probability of a state depends only on the previous
state, is combined with a Monte Carlo simulation method, see e.g. (:cite:`MCMC1`, :cite:`MCMC2`). More specifically, a Markov Chain is
built and sampled from whose stationary distribution is the target distribution :math:`p(x)`.  For instance, the
Metropolis-Hastings (MH) algorithm goes as follows:

- initialize with a seed sample :math:`x_{0}`
- walk the chain: for :math:`k=0,...` do:
   * sample candidate :math:`x^{\star} \sim Q(\cdot \vert x_{k})` for a given Markov transition probability :math:`Q`
   * accept candidate (set :math:`x_{k+1}=x^{\star}`) with probability :math:`\alpha(x^{\star} \vert x_{k})`, otherwise propagate last sample :math:`x_{k+1}=x_{k}`.

.. math:: \alpha(x^{\star} \vert x_{k}):= \min \left\{ \frac{\tilde{p}(x^{\star})}{\tilde{p}(x)}\cdot \frac{Q(x \vert x^{\star})}{Q(x^{\star} \vert x)}, 1 \right\}

The transition probability :math:`Q` is chosen by the user (see input `proposal` of the MH algorithm, and careful
attention must be given to that choice as it plays a major role in the accuracy and efficiency of the algorithm. The
following figure shows samples accepted (blue) and rejected (red) when trying to sample from a 2d Gaussian distribution
using MH, for different scale parameters of the proposal distribution. If the scale is too small, the space is not well
explored; if the scale is too large, many candidate samples will be rejected, yielding a very inefficient algorithm.
As a rule of thumb, an acceptance rate of 10\%-50\% could be targeted.

.. image:: ../../_static/SampleMethods_MCMC_samples.png
   :scale: 40 %
   :alt: IS weighted samples
   :align: center

Finally, samples from the target distribution will be generated only when the chain has converged to its stationary
distribution, after a so-called burn-in period. Thus the user would often reject the first few samples (see input
`nburn`). Also, the chain yields correlated samples; thus to obtain i.i.d. samples from the target distribution,
the user should keep only one out of n samples (see input `jump`). This means that the code will perform in total
:code:`burn_length + jump * N` evaluations of the target pdf to yield N i.i.d. samples from the target distribution (for the MH
algorithm with a single chain).

The parent class for all MCMC algorithms is the :class:`.MCMC` class, which defines the inputs that are common to all
MCMC algorithms, along with the :meth:`run` method that is being called to run the chain. Any given MCMC algorithm is a
child class of MCMC that overwrites the main :meth:`run_one_iteration` method.

MCMC Class
^^^^^^^^^^^

Methods
~~~~~~~~~~~~~~~~~~
.. autoclass:: UQpy.sampling.mcmc.MCMC
   :exclude-members: __init__
   :members: run, run_one_iteration

Attributes
~~~~~~~~~~~~~~~~~~
.. autoattribute:: UQpy.sampling.mcmc.MCMC.evaluate_log_target
.. autoattribute:: UQpy.sampling.mcmc.MCMC.evaluate_log_target_marginals
.. autoattribute:: UQpy.sampling.mcmc.MCMC.samples
.. autoattribute:: UQpy.sampling.mcmc.MCMC.log_pdf_values
.. autoattribute:: UQpy.sampling.mcmc.MCMC.nsamples
.. autoattribute:: UQpy.sampling.mcmc.MCMC.nsamples_per_chain
.. autoattribute:: UQpy.sampling.mcmc.MCMC.iterations_number


Examples
~~~~~~~~~~~~~~~~~~
.. toctree::

   Markov Chain Monte Carlo Sampling Examples <../../auto_examples/sampling/mcmc/index>


List of MCMC algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

    Metropolis Hastings <mh>
    Modified Metropolis Hastings <mmh>
    DRAM <dram>
    DREAM <dream>
    Stretch <stretch>


Adding New MCMC Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to add a new MCMC algorithm, a user must create a child class of :meth:`.MCMC`, and overwrite the
:meth:`run_one_iteration` method that propagates all the chains forward one iteration. Such a new class may use any
number of additional inputs compared to the :class:`.MCMC` base class. The reader is encouraged to have a look at the
:class:`.MetropolisHastings` class and its code to better understand how a particular algorithm should fit the general framework.

A useful note is that the user has access to a number of useful attributes / utility methods as the algorithm proceeds, such as:

* the attribute :meth:`evaluate_log_target` (and possibly :meth:`evaluate_log_target_marginals` if marginals were provided) is created at initialization. It is a callable that simply evaluates the log-pdf of the target distribution at a given point **x**. It can be called within the code of a new sampler as ``log_pdf_value = self.evaluate_log_target(x)``.
* the :py:attr:`samples_number` and :py:attr:`samples_number_per_chain` attributes indicate the number of samples that have been stored up to the current iteration (i.e., they are updated dynamically as the algorithm proceeds),
* the :py:attr:`samples` attribute contains all previously stored samples. Cautionary note: :code:`self.samples` also contains trailing zeros, for samples yet to be stored, thus to access all previously stored samples at a given iteration the user must call ``self.samples[:self.nsamples_per_chain]``, which will return an :class:`numpy.ndarray` of size :code:`(self.nsamples_per_chain, self.nchains, self.dimension)` ,
* the :py:attr:`log_pdf_values` attribute contains all previously stored log target values. Same cautionary note as above,
* the :meth:`_update_acceptance_rate` method updates the :py:attr:`acceptance_rate` attribute of the sampler, given a (list of) boolean(s) indicating if the candidate state(s) were accepted at a given iteration,
* the :meth:`_check_methods_proposal` method checks whether a given proposal is adequate (i.e., has :meth:`rvs` and :meth:`log_pdf`/:meth:`pdf` methods).


