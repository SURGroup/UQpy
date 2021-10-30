Subset Simulation
-------------------

In the subset simulation method [3]_ the probability of failure :math:`P_f`  is approximated by a product of probabilities
of more frequent events. That is, the failure event :math:`G = \{\textbf{x} \in \mathbb{R}^n:G(\textbf{x}) \leq 0\}`,
is expressed as the of union of `M` nested intermediate events :math:`G_1,G_2,\cdots,G_M` such that
:math:`G_1 \supset G_2 \supset \cdots \supset G_M`, and :math:`G = \cap_{i=1}^{M} G_i`. The intermediate failure events
are defined as :math:`G_i=\{G(\textbf{x})\le b_i\}`, where :math:`b_1>b_2>\cdots>b_i=0` are positive thresholds selected
such that each conditional probability :math:`P(G_i | G_{i-1}), ~i=2,3,\cdots,M-1` equals a target probability value
:math:`p_0`. The probability of failure :math:`P_f` is estimated as:

.. math:: P_f = P\left(\cap_{i=1}^M G_i\right) = P(G_1)\prod_{i=2}^M P(G_i | G_{i-1})

where the probability :math:`P(G_1)` is computed through Monte Carlo simulations. In order to estimate the conditional
probabilities :math:`P(G_i|G_{i-1}),~j=2,3,\cdots,M` generation of Markov Chain Monte Carlo (MCMC) samples from the
conditional pdf :math:`p_{\textbf{U}}(\textbf{u}|G_{i-1})` is required. In the context of subset simulation, the Markov
chains are constructed through a two-step acceptance/rejection criterion. Starting from a Markov chain state
:math:`\textbf{x}` and a proposal distribution :math:`q(\cdot|\textbf{x})`, a candidate sample :math:`\textbf{w}` is
generated. In the first stage, the sample :math:`\textbf{w}` is accepted/rejected with probability

.. math:: \alpha=\min\bigg\{1, \frac{p(\textbf{w})q(\textbf{x}|\textbf{w})}{p(\textbf{x})q(\textbf{w}|\textbf{x})}\bigg\}

and in the second stage is accepted/rejected based on whether the sample belongs to the failure region :math:`G_j`.
:class:`.SubSetSimulation` can be used with any of the available (or custom) :class:`.MCMC` classes in the
:py:mod:`sampling` module.

SubsetSimulation Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.reliability.SubsetSimulation
    :members:

.. [3] S.K.  Au  and  J.L.  Beck. "Estimation  of  small  failure  probabilities  in  high  dimensions  by  subset  simulation", Probabilistic  Engineering Mechanics, 16(4):263–277, 2001.