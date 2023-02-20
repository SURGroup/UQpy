Tempering MCMC
~~~~~~~~~~~~~~

Tempering MCMC algorithms aim at sampling from a target distribution :math:`p_1(x)` of the form
:math:`p_1(x)=\frac{q_1(x)p_0(x)}{Z_1}` where the factor :math:`q_1(x)` and reference distribution
:math:`p_0(x)` can be evaluated. Additionally, these algorithms return an estimate of the normalization
constant :math:`Z_1=\int{q_{1}(x) p_{0}(x)dx}`.

The algorithms sample from a sequence of intermediate densities
:math:`p_{\beta}(x) \propto q_{\beta}(x) p_{0}(x)` for values of the parameter :math:`\beta` between 0 and 1
(:math:`\beta=\frac{1}{T}` where :math:`T` is sometimes called the temperature, :math:`q_{\beta}(x)` is referred to as the intermediate factor associated with tempering parameter :math:`\beta`).
Setting :math:`\beta = 1` equates sampling from the target, while :math:`\beta \rightarrow 0` samples from the
reference distribution.

Parallel tempering samples from all distributions simultaneously, and the tempering parameters :math:`0 < \beta_1 < \beta_2 < \cdots < \beta_{N} \leq 1` must be
chosen in advance by the user. Sequential tempering on the other hand samples from the various distributions sequentially, starting
from the reference distribution, and the tempering parameters are selected adaptively by the algorithm.

The :class:`.TemperingMCMC` base class defines inputs that are common to parallel and sequential tempering:

.. autoclass:: UQpy.sampling.mcmc.tempering_mcmc.TemperingMCMC
    :members:
    :exclude-members: run, evaluate_normalization_constant

Parallel Tempering
^^^^^^^^^^^^^^^^^^^^

This algorithm (see e.g. :cite:`PTMCMC1` for theory about parallel tempering) runs the chains sampling from the various tempered distributions simultaneously. Periodically during the
run, the different temperatures swap members of their ensemble in a way that preserves detailed balance. The chains
closer to the reference chain (hot chains) can sample from regions that have low probability under the target and
thus allow a better exploration of the parameter space, while the cold chains can better explore regions of high
likelihood.

In parallel tempering, the normalizing constant :math:`Z_1` is evaluated via thermodynamic integration (:cite:`PTMCMC2`). Defining
the potential function :math:`U_{\beta}(x)=\frac{\partial \log{q_{\beta}(x)}}{\partial \beta}`, then

:math:`\ln{Z_1} = \log{Z_0} + \int_{0}^{1} E_{x \sim p_{\beta}} \left[ U_{\beta}(x) \right] d\beta`

where the expectations are approximated via MC sampling using saved samples from the intermediate distributions. A trapezoidal rule is used for integration.

The :class:`.ParallelTemperingMCMC` class is imported using the following command:

>>> from UQpy.sampling.mcmc.tempering_mcmc.ParallelTemperingMCMC import ParallelTemperingMCMC

.. autoclass:: UQpy.sampling.mcmc.ParallelTemperingMCMC
    :members: run, evaluate_normalization_constant

Sequential Tempering
^^^^^^^^^^^^^^^^^^^^

This algorithm (first introduced in :cite:`STMCMC_ChingChen`) samples from a series of intermediate targets that are each tempered versions of the final/true
target. In going from one intermediate distribution to the next, the existing samples are resampled according to
some weights (similar to importance sampling). To ensure that there aren't a large number of duplicates, the
resampling step is followed by a short (or even single-step) Metropolis Hastings run that disperses the samples while
remaining within the correct intermediate distribution. The final intermediate target is the required target distribution,
and the samples following this distribution are the required samples.

The normalization constant :math:`Z_1` is estimated as the product of the normalized sums of the resampling weights for
each intermediate distribution, i.e. if :math:`w_{\beta_j}(x_{j_i})` is the resampling weight corresponding to tempering
parameter :math:`\beta_j`, calculated for the i-th sample for the intermediate distribution associated with :math:`\beta_j`,
then :math:`Z_1 = \prod_{j=1}^{N} \ [ \sum_{i=i}^{\text{nsamples}} \ ]`. The Coefficient of Variance (COV) for this
estimator is also given in :cite:`STMCMC_ChingChen`.

The :class:`.SequentialTemperingMCMC` class is imported using the following command:

>>> from UQpy.sampling.mcmc.tempering_mcmc.SequentialTemperingMCMC import SequentialTemperingMCMC

.. autoclass:: UQpy.sampling.mcmc.tempering_mcmc.SequentialTemperingMCMC
    :members:

Examples
~~~~~~~~~~~~~~~~~~

.. toctree::

   Tempering Examples <../../auto_examples/sampling/tempering/index>