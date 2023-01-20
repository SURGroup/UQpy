Sobol Sensitivity indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These examples serve as a guide for using the Sobol sensitivity module. They have been taken from various papers to enable validation of the implementation and have been referenced accordingly.

Single output models
======================
We demonstrate the computation of the Sobol indices for models with a single output using the following examples:

1. **Additive function**

    This is a beginner-friendly example for introducing Sobol indices. The function is a linear combination of two inputs which produces a scalar output.

2. **Ishigami function**

    The Ishigami function is a non-linear, non-monotonic function that is commonly used to benchmark uncertainty and senstivity analysis methods.

3. **Sobol function** 

    The Sobol function is non-linear function that is commonly used to benchmark uncertainty 
    and senstivity analysis methods. Unlike the Ishigami function which has 3 input 
    variables, the Sobol function can have any number of input variables (see [2]_).

Multiple output models
========================

We demonstrate the computation of the Sobol indices for models with multiple outputs using the following example:

1. **Mechanical oscillator ODE**

    The Sobol indices are computed for a mechanical oscillator governed by a second-order differential equation [1]_. The model outputs the displacement of the oscillator for a given time period. Here the sensitivity of the model parameters are computed at each point in time (see [1]_).

.. [1] Gamboa F, Janon A, Klein T, Lagnoux A, others. Sensitivity analysis for multidimensional and functional outputs. Electronic journal of statistics 2014; 8(1): 575-603.

.. [2] Saltelli, A. (2002). Making best use of model evaluations to compute  indices.
