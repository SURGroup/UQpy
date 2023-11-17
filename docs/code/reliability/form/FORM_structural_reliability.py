"""

3. FORM - Structural Reliability
==============================================

The benchmark problem is a simple structural reliability problem (example 7.1 in :cite:`FORM_XDu`)
defined in a two-dimensional parameter space consisting of a resistance :math:`R` and a stress :math:`S`. The failure
happens when the stress is higher than the resistance, leading to the following limit-state function:

.. math:: \\textbf{X}=\{R, S\}

.. math:: g(\\textbf{X}) = R - S

The two random variables are independent  and  distributed
according to:

.. math:: R \sim N(200, 20)

.. math:: S \sim N(150, 10)
"""

# %% md
#
# Initially we have to import the necessary modules.

# %%

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from UQpy.distributions import Normal
from UQpy.reliability import FORM
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel


# %% md
#
# Next, we initialize the :code:`RunModel` object.
# The `local_pfn.py <https://github.com/SURGroup/UQpy/tree/master/docs/code/reliability/sorm>`_ file can be found on
# the UQpy GitHub. It contains a simple function :code:`example1` to compute the difference between the resistence and the
# stress.

# %%

model = PythonModel(model_script='local_pfn.py', model_object_name="example1")
runmodel_object = RunModel(model=model)

# %% md
#
# Now we can define the resistence and stress distributions that will be passed into :code:`FORM`.
# Along with the distributions, :code:`FORM` takes in the previously defined :code:`runmodel_object` and tolerances
# for convergences. Since :code:`tolerance_gradient` is not specified in this example, it is not considered.

# %%

distribution_resistance = Normal(loc=200., scale=20.)
distribution_stress = Normal(loc=150., scale=10.)
form = FORM(distributions=[distribution_resistance, distribution_stress], runmodel_object=runmodel_object,
            tolerance_u=1e-5, tolerance_beta=1e-5)
# %% md
#
# With everything defined we are ready to run the first-order reliability method and print the results.
# The analytic solution to this problem is :math:`\textbf{u}^*=(-2, 1)` with a reliability index of
# :math:`\beta_{HL}=2.2361` and a probability of failure :math:`P_{f, \text{form}} = \Phi(-\beta_{HL}) = 0.0127`

# %%

form.run()
print('Design point in standard normal space:', form.design_point_u)
print('Design point in original space:', form.design_point_x)
print('Hasofer-Lind reliability index:', form.beta)
print('FORM probability of failure:', form.failure_probability)
print('FORM record of the function gradient:', form.state_function_gradient_record)

# %% md
#
# This problem can be visualized in the following plots that show the FORM results in both :math:`\textbf{X}` and
# :math:`\textbf{U}` space.

# %%

def multivariate_gaussian(pos, mu, sigma):
    """Supporting function"""
    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi) ** n * sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)
    return np.exp(-fac / 2) / N


N = 60
XX = np.linspace(150, 250, N)
YX = np.linspace(120, 180, N)
XX, YX = np.meshgrid(XX, YX)

XU = np.linspace(-3, 3, N)
YU = np.linspace(-3, 3, N)
XU, YU = np.meshgrid(XU, YU)


# %% md
#
# Define the mean vector and covariance matrix in the original :math:`\textbf{X}` space and the standard normal
# :math:`\textbf{U}` space.

# %%
mu_X = np.array([distribution_resistance.parameters['loc'], distribution_stress.parameters['loc']])
sigma_X = np.array([[distribution_resistance.parameters['scale']**2, 0],
                    [0, distribution_stress.parameters['scale']**2]])

mu_U = np.array([0, 0])
sigma_U = np.array([[1, 0],
                    [0, 1]])

# Pack X and Y into a single 3-dimensional array for the original space
posX = np.empty(XX.shape + (2,))
posX[:, :, 0] = XX
posX[:, :, 1] = YX
ZX = multivariate_gaussian(posX, mu_X, sigma_X)

# Pack X and Y into a single 3-dimensional array for the standard normal space
posU = np.empty(XU.shape + (2,))
posU[:, :, 0] = XU
posU[:, :, 1] = YU
ZU = multivariate_gaussian(posU, mu_U, sigma_U)

# %% md
#
# Plot the :code:`FORM` solution in the original :math:`\textbf{X}` space and the standard normal :math:`\text{U}`
# space.

# %%
fig, ax = plt.subplots()
ax.contour(XX, YX, ZX,
           levels=20)
ax.plot([0, 200], [0, 200],
        color='black', linewidth=2, label='$G(R,S)=R-S=0$', zorder=1)
ax.scatter(mu_X[0], mu_X[1],
           color='black', s=64, label='Mean $(\mu_R, \mu_S)$')
ax.scatter(form.design_point_x[0][0], form.design_point_x[0][1],
           color='tab:orange', marker='*', s=100, label='Design Point', zorder=2)
ax.set(xlabel='Resistence $R$', ylabel='Stress $S$', xlim=(145, 255), ylim=(115, 185))
ax.set_title('Original $X$ Space ')
ax.set_aspect('equal')
ax.legend(loc='lower right')

fig, ax = plt.subplots()
ax.contour(XU, YU, ZU,
           levels=20, zorder=1)
ax.plot([0, -3], [5, -1],
        color='black', linewidth=2, label='$G(U_1, U_2)=0$', zorder=2)
ax.arrow(0, 0, form.design_point_u[0][0], form.design_point_u[0][1],
         color='tab:blue', length_includes_head=True, width=0.05, label='$\\beta=||u^*||$', zorder=2)
ax.scatter(form.design_point_u[0][0], form.design_point_u[0][1],
           color='tab:orange', marker='*', s=100, label='Design Point $u^*$', zorder=2)
ax.set(xlabel='$U_1$', ylabel='$U_2$', xlim=(-3, 3), ylim=(-3, 3))
ax.set_aspect('equal')
ax.set_title('Standard Normal $U$ Space')
ax.legend(loc='lower right')

plt.show()
