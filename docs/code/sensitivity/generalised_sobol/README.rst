Generalised Sobol Sensitivity indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These examples serve as a guide for using the GSI sensitivity module. They have been taken from various papers to enable validation of the implementation and have been referenced accordingly.

1. **Mechanical oscillator ODE**

    The GSI sensitivity indices are computed for a mechanical oscillator governed by a second-order differential equation [1]_. The model outputs the displacement of the oscillator for a given time period. Unlike the pointwise-in-time Sobol indices, which provide the sensitivity of the model parameters at each point in time, the GSI indices summarise the sensitivities of the model parameters over the entire time period.

2. **Toy example**
    
    The GSI sensitivity indices are computed for a toy model whose analytical solution is given in [1]_.

.. [1] Gamboa F, Janon A, Klein T, Lagnoux A, others.  Sensitivity analysis for multidimensional and functional outputs. Electronic journal of statistics 2014; 8(1): 575-603.