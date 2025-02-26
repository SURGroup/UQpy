"""

Physics-Informed PCE: Uncertainty Quantification of Euler Beam
======================================================================

In this example, we use Physics-informed Polynomial Chaos Expansion for approximation and UQ of Euler beam equation.
"""

# %% md
#
# Import necessary libraries.

# %%

from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *

# load PC^2
from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import TotalDegreeBasis
from UQpy.surrogates.polynomial_chaos.physics_informed.ConstrainedPCE import ConstrainedPCE
from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData
from UQpy.surrogates.polynomial_chaos.physics_informed.PdePCE import PdePCE
from UQpy.surrogates.polynomial_chaos.physics_informed.Utilities import *
from UQpy.surrogates.polynomial_chaos.physics_informed.ReducedPCE import ReducedPCE


# %% md
#
# We then define functions used for construction of Physically Constrained PCE (PC :math:`^2`)

# %%

# Definition of PDE/ODE
def pde_func(standardized_sample, pce):
    der_order = 4
    deriv_pce = derivative_basis(standardized_sample, pce, derivative_order=der_order, leading_variable=0) * ((2 / 1) ** der_order)

    pde_basis = deriv_pce

    return pde_basis


# Definition of the source term
def pde_res(standardized_sample):
    load_s = load(standardized_sample)

    return -load_s[:, 0]


def load(standardized_sample):
    return const_load(standardized_sample)


def const_load(standardized_sample):
    l = (1 + (1 + standardized_sample[:, 1]) / 2).reshape(-1, 1)
    return l


# Definition of the function for sampling of boundary conditions
def bc_sampling(nsim=1000):
    # BC sampling

    nsim_half = round(nsim / 2)
    sample = np.zeros((nsim, 2))
    real_ogrid_1d = ortho_grid(nsim_half, 1, 0.0, 1.0)[:, 0]

    sample[:nsim_half, 0] = np.zeros(nsim_half)
    sample[:nsim_half, 1] = real_ogrid_1d

    sample[nsim_half:, 0] = np.ones(nsim_half)
    sample[nsim_half:, 1] = real_ogrid_1d

    return sample


# define sampling and evaluation of BC for estimation of error
def bc_res(nsim, pce):
    physical_sample= np.zeros((2, 2))
    physical_sample[1, 0] = 1
    standardized_sample = polynomial_chaos.Polynomials.standardize_sample(physical_sample, pce.polynomial_basis.distributions)

    der_order = 2
    deriv_pce = np.sum(
        derivative_basis(standardized_sample, pce, derivative_order=der_order, leading_variable=0) *
        ((2 / 1) ** der_order) * np.array(pce.coefficients).T, axis=1)

    return deriv_pce


# Definition of the reference solution for an error estimation
def ref_sol(physical_coordinate, q):
    return (q + 1) * (-(physical_coordinate ** 4) / 24 + physical_coordinate ** 3 / 12 - physical_coordinate / 24)


# %% md
#
# Beam equation is parametrized by one deterministic input variable (coordinate :math:`x`) and
# random load intensity :math:`q`, both modeled as Uniform random variables.

# number of input variables

nrand = 1
nvar = 1 + nrand

#   definition of a joint probability distribution
dist1 = Uniform(loc=0, scale=1)
dist2 = Uniform(loc=0, scale=1)
marg = [dist1, dist2]
joint = JointIndependent(marginals=marg)

# %% md
#
# The physical domain is defined by :math:`x \in [0,1]`


# define geometry of physical space
geometry_xmin = [0]
geometry_xmax = [1]

# %% md
#
# Boundary conditions are prescribed as
#
# .. math:: \frac{d^2 u(0,q)}{d x^2}=0, \qquad \frac{d^2 u(1,q)}{d x^2}=0, \qquad u(0,q)=0, \qquad  u(1,q)=0

# number of BC samples
nbc = 2 * 10

# derivation orders of prescribed BCs
der_orders = [0, 2]
# normals associated to precribed BCs
bc_normals = [0, 0]
# sampling of BC points

bc_xtotal = bc_sampling(20)
bc_ytotal = np.zeros(len(bc_xtotal))

bc_x = [bc_xtotal, bc_xtotal]
bc_y = [bc_ytotal, bc_ytotal]

# %% md
#
# Here we construct an object containing general physical information (geometry, boundary conditions)
pde_data = PdeData(geometry_xmax, geometry_xmin, der_orders, bc_normals, bc_x, bc_y)

# %% md
#
# Further we construct an object containing PDE physical data and PC :math:`^2` definitions of PDE

pde_pce = PdePCE(pde_data, pde_func, pde_source=pde_res, boundary_conditions_evaluate=bc_res)

# %% md
#
# Get dirichlet boundary conditions from PDE data object that are further used for construction of initial PCE object.

dirichlet_bc = pde_data.dirichlet
x_train = dirichlet_bc[:, :-1]
y_train = dirichlet_bc[:, -1]

# %% md
#
# Finally, PC :math:`^2` is constructed using Karush-Kuhn-Tucker normal equations solved by ordinary least squares and
# Least Angle Regression.

least_squares = LeastSquareRegression()
p = 9

PCEorder = p
polynomial_basis = TotalDegreeBasis(joint, PCEorder, hyperbolic=1)

#  create initial PCE object containing basis, regression method and dirichlet BC
initpce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
initpce.set_data(x_train, y_train)

# construct a PC^2 object combining pde_data, pde_pce and initial PCE objects
pcpc = ConstrainedPCE(pde_data, pde_pce, initpce)
# get coefficients of PC^2 by least angle regression
pcpc.lar()

# get coefficients of PC^2 by ordinary least squares
pcpc.ols()

# evaluate errors of approximations
real_ogrid = ortho_grid(100, nvar, 0.0, 1.0)
yy_val_pce = pcpc.lar_pce.predict(real_ogrid).flatten()
yy_val_pce_ols = pcpc.initial_pce.predict(real_ogrid).flatten()
yy_val_true = ref_sol(real_ogrid[:, 0], real_ogrid[:, 1]).flatten()

err = np.abs(yy_val_pce - yy_val_true)
tot_err = np.sum(err)
print('\nTotal approximation error by PC^2-LAR: ', tot_err)

err_ols = np.abs(yy_val_pce_ols - yy_val_true)
tot_err_ols = np.sum(err_ols)
print('Total approximation error by PC^2-OLS: ', tot_err_ols)

# %% md
#
# Once the PC :math:`^2` is constructed, we use ReducedPce class to filter out influence of deterministic variable in UQ

reduced_pce = ReducedPCE(pcpc.lar_pce, n_deterministic=1)

# %% md
#
# ReducedPce is used for estimation of local statistical moments and quantiles :math:`\pm2\sigma`
coeff_res = []
var_res = []
mean_res = []
vartot_res = []
lower_quantiles_modes = []
upper_quantiles_modes = []

n_derivations = 4
sigma_mult = 2
beam_x = np.arange(0, 101) / 100

for x in beam_x:
    mean = np.zeros(n_derivations + 1)
    var = np.zeros(n_derivations + 1)
    variances = np.zeros((n_derivations + 1, nrand))
    lq = np.zeros((1 + n_derivations, nrand))
    uq = np.zeros((1 + n_derivations, nrand))
    for d in range(1 + n_derivations):
        if d == 0:
            coeff = (reduced_pce.evaluate_coordinate(np.array(x), return_coefficients=True))
        else:
            coeff = reduced_pce.derive_coordinate(np.array(x), derivative_order=d, leading_variable=0,
                                                  return_coefficients=True,
                                                  derivative_multiplier=transformation_multiplier(pde_data, 0, d))
        mean[d] = coeff[0]
        var[d] = np.sum(coeff[1:] ** 2)
        variances[d, :] = reduced_pce.variance_contributions(coeff)

        for e in range(nrand):
            lq[d, e] = mean[d] + sigma_mult * np.sqrt(np.sum(variances[d, :e + 1]))
            uq[d, e] = mean[d] - sigma_mult * np.sqrt(np.sum(variances[d, :e + 1]))

    lower_quantiles_modes.append(lq)
    upper_quantiles_modes.append(uq)
    var_res.append(variances)
    mean_res.append(mean)
    vartot_res.append(var)

mean_res = np.array(mean_res)
vartot_res = np.array(vartot_res)
var_res = np.array(var_res)
lower_quantiles_modes = np.array(lower_quantiles_modes)
upper_quantiles_modes = np.array(upper_quantiles_modes)

# %% md
#
# PC :math:`^2` and corresponding Reduced PCE can be further used for local UQ. Obtained results are depicted in figure.

# plot results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 4, figsize=(14.5, 3))
colors = ['black', 'blue', 'green', 'red']

for d in range(2, 1 + n_derivations):
    ax[d - 1].plot(beam_x, mean_res[:, d], color=colors[d - 1])
    ax[d - 1].plot(beam_x, mean_res[:, d] + sigma_mult * np.sqrt(vartot_res[:, d]), color=colors[d - 1])
    ax[d - 1].plot(beam_x, mean_res[:, d] - sigma_mult * np.sqrt(vartot_res[:, d]), color=colors[d - 1])

    ax[d - 1].plot(beam_x, lower_quantiles_modes[:, d, 0], '--', alpha=0.7, color=colors[d - 1])
    ax[d - 1].plot(beam_x, upper_quantiles_modes[:, d, 0], '--', alpha=0.7, color=colors[d - 1])
    ax[d - 1].fill_between(beam_x, lower_quantiles_modes[:, d, 0], upper_quantiles_modes[:, d, 0],
                           facecolor=colors[d - 1], alpha=0.05)

    ax[d - 1].plot(beam_x, np.zeros(len(beam_x)), color='black')

ax[0].plot(beam_x, mean_res[:, 0], color=colors[0])
ax[0].plot(beam_x, mean_res[:, 0] + sigma_mult * np.sqrt(vartot_res[:, 0]), color=colors[0])
ax[0].plot(beam_x, mean_res[:, 0] - sigma_mult * np.sqrt(vartot_res[:, 0]), color=colors[0])

ax[0].plot(beam_x, lower_quantiles_modes[:, 0, 0], '--', alpha=0.7, color=colors[0])
ax[0].plot(beam_x, upper_quantiles_modes[:, 0, 0], '--', alpha=0.7, color=colors[0])
ax[0].fill_between(beam_x, lower_quantiles_modes[:, 0, 0], upper_quantiles_modes[:, 0, 0],
                   facecolor=colors[0], alpha=0.05)

ax[0].plot(beam_x, np.zeros(len(beam_x)), color='black')

ax[3].invert_yaxis()
ax[1].invert_yaxis()

ax[3].set_title(r'Load $\frac{\partial^4 w}{\partial x^4}$', y=1.04)
ax[2].set_title(r'Shear Force $\frac{\partial^3 w}{\partial x^3}$', y=1.04)
ax[1].set_title(r'Bending Moment $\frac{\partial^2 w}{\partial x^2}$', y=1.04)
ax[0].set_title(r'Deflection $w$', y=1.04)

for axi in ax.flatten():
    axi.set_xlabel(r'$x$')
plt.show()
