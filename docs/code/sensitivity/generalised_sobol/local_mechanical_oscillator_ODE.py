"""

Auxiliary file
==============================================

"""

import numpy as np
from scipy.integrate import solve_ivp


def mech_oscillator(input_parameters):
    """
    We have the second order differential equation:

    .. math::

        m \ddot{x} + c \dot{x} + k x = 0

    with initial conditions: :math: `x(0) = \ell`, :math: `\dot{x}(0) = 0`.

    where, for example  :math: `m \sim \mathcal{U}(10, 12)`,
                        :math: `c \sim \mathcal{U}(0.4, 0.8)`
                        :math: `k \sim \mathcal{U}(70, 90)`
                        :math: `\ell \sim \mathcal{U}(-1, -0.25)`.


    References
    ----------

    .. [1] Gamboa F, Janon A, Klein T, Lagnoux A, others .
        Sensitivity analysis for multidimensional and functional outputs.
        Electronic journal of statistics 2014; 8(1): 575-603.

    """

    # unpack the input parameters
    m, c, k, l = input_parameters[0]

    # intial conditions
    x_0 = l
    v_0 = 0

    # time points
    t_0 = 0
    t_f = 40
    dt = 0.05
    n_t = int((t_f - t_0) / dt)
    T = np.linspace(t_0, t_f, n_t)

    def ODE(t, y):
        """
        The ODE system.
        """
        return np.array([y[1], -(k / m) * y[0] - (c / m) * y[1]])

    # solve the ODE
    sol = solve_ivp(ODE, [t_0, t_f], [x_0, v_0], method="RK45", t_eval=T)

    return sol.y[0]
