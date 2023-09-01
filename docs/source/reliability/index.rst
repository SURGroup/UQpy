Reliability
===========

Reliability of a system refers to the assessment of its probability of failure (i.e the system no longer satisfies some
performance measure), given the model uncertainty in the structural, environmental and load parameters. Given a vector
of random variables :math:`\textbf{X}=[X_1, X_2, \ldots, X_n]^T \in \mathcal{D}_\textbf{X}\subset \mathbb{R}^n`, where
:math:`\mathcal{D}_\textbf{X}` is the domain of interest and :math:`f_{\textbf{X}}(\textbf{x})` is its joint probability density
function, then the probability that the system will fail is defined as


.. math:: P_f =\mathbb{P}(g(\textbf{X}) \leq 0) = \int_{\mathcal{D}_f} f_{\textbf{X}}(\textbf{x})d\textbf{x} = \int_{\{\textbf{X}:g(\textbf{X})\leq 0 \}} f_{\textbf{X}}(\textbf{x})d\textbf{x}


where :math:`g(\textbf{X})` is the so-called performance function and :math:`\mathcal{D}_f=\{\textbf{X}:g(\textbf{X})\leq 0 \}` is the failure domain.
The reliability problem is often formulated in the
standard normal space :math:`\textbf{U}\sim \mathcal{N}(\textbf{0}, \textbf{I}_n)`, which means that a nonlinear
isoprobabilistic  transformation from the generally non-normal parameter space
:math:`\textbf{X}\sim f_{\textbf{X}}(\cdot)` to the standard normal space is required (see the :py:mod:`.transformations` module).
The performance function in the standard normal space is denoted :math:`G(\textbf{U})`. :py:mod:`.UQpy` does not require this
transformation and can support reliability analysis for problems with arbitrarily distributed parameters.


This module contains functionality for all reliability methods supported in :py:mod:`UQpy`.
The module currently contains the following classes:

- :class:`.TaylorSeries`: Class to perform reliability analysis using First Order reliability Method (:class:`FORM`) and Second Order
  Reliability Method (:class:`SORM`).
- :class:`.SubsetSimulation`: Class to perform reliability analysis using subset simulation.



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reliability

    Subset Simulation <subset>
    Taylor Series <taylor_series>