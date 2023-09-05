Subset Simulation
-------------------

In the subset simulation method :cite:`SubsetSimulation` the probability of failure :math:`P_f`  is approximated by a product of probabilities
of more frequent events. That is, the failure event :math:`G = \{\textbf{X} \in \mathbb{R}^n:g(\textbf{X}) \leq 0\}`,
is expressed as the of union of `M` nested intermediate events :math:`G_1,G_2,\cdots,G_M` such that
:math:`G_1 \supset G_2 \supset \cdots \supset G_M`, and :math:`G = \cap_{i=1}^{M} G_i`. The intermediate failure events
are defined as :math:`G_i=\{g(\textbf{X})\le b_i\}`, where :math:`b_1>b_2>\cdots>b_M=0` are non-negative thresholds selected
such that each conditional probability :math:`P(G_{i+1} | G_{i}),\ i=1,2,\cdots,M-1` equals a target probability value
:math:`p_0`. The probability of failure :math:`P_f` is estimated as:

.. math:: P_f = P\left(\bigcap_{i=1}^M G_i\right) = P(G_1)\prod_{i=1}^{M-1} P(G_{i+1} | G_{i})

where the probability :math:`P(G_1)` is computed through Monte Carlo simulations. In order to estimate the conditional
probabilities :math:`P(G_{i+1}|G_i),~i=1,2,\cdots,M-1` generation of Markov Chain Monte Carlo (MCMC) samples from the
conditional pdf :math:`p_{\textbf{X}}(\textbf{x}|G_i)` is required. In the context of subset simulation, the Markov
chains are constructed through a two-step acceptance/rejection criterion. Starting from a Markov chain state
:math:`\textbf{X}` and a proposal distribution :math:`q(\cdot|\textbf{X})`, a candidate sample :math:`\textbf{W}` is
generated. In the first stage, the sample :math:`\textbf{W}` is accepted/rejected with probability

.. math:: \alpha=\min\bigg\{1, \frac{p_\textbf{X}(\textbf{w})q(\textbf{x}|\textbf{W})}{p_\textbf{X}(\textbf{x})q(\textbf{w}|\textbf{X})}\bigg\}

and in the second stage is accepted/rejected based on whether the sample belongs to the failure region :math:`G_i`.
:class:`.SubsetSimulation` can be used with any of the available (or custom) :class:`.MCMC` classes in the
:py:mod:`Sampling` module.

The :class:`.SubsetSimulation` class is imported using the following command:

>>> from UQpy.reliability.SubsetSimulation import SubsetSimulation

Methods
"""""""

.. autoclass:: UQpy.reliability.SubsetSimulation

Attributes
""""""""""

.. autoattribute:: UQpy.reliability.SubsetSimulation.dependent_chains_CoV
.. autoattribute:: UQpy.reliability.SubsetSimulation.failure_probability
.. autoattribute:: UQpy.reliability.SubsetSimulation.independent_chains_CoV
.. autoattribute:: UQpy.reliability.SubsetSimulation.performance_function_per_level
.. autoattribute:: UQpy.reliability.SubsetSimulation.performance_threshold_per_level
.. autoattribute:: UQpy.reliability.SubsetSimulation.samples

Examples
""""""""""

.. toctree::

   Subset Simulation Examples <../auto_examples/reliability/subset_simulation/index>
