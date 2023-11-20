"""

Physics-Informed PCE: Deterministic Wave Equation
======================================================================

In this example, we approximate a deterministic Wave Equation by Physics-informed Polynomial Chaos Expansion.
"""

# %% md
#
# Import necessary libraries.

# %%


from UQpy.surrogates import *
from UQpy.distributions import Uniform, JointIndependent
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import *
from UQpy.sampling import LatinHypercubeSampling

# load PC^2
from UQpy.surrogates.polynomial_chaos.physics_informed.ConstrainedPCE import ConstrainedPCE
from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData
from UQpy.surrogates.polynomial_chaos.physics_informed.PdePCE import PdePCE
from UQpy.surrogates.polynomial_chaos.physics_informed.Utilities import *


# %% md
#
# We then define functions used for construction of Physically Constrained PCE (PC :math:`^2`)

# %%

# Definition of PDE in context of PC^2
def pde_func(s, pce):
    # function inputs must be input random vector in standardized space and pce object

    # partial derivation according to the first input variable x (second order)
    # derived basis must be multiplied by constant reflecting the difference between physical and standardized space
    deriv_0_pce = derivative_basis(s, pce, derivative_order=2, leading_variable=0) * transformation_multiplier(pde_data,
                                                                                                               leading_variable=0,
                                                                                                               derivation_order=2)
    deriv_1_pce = derivative_basis(s, pce, derivative_order=2, leading_variable=1) * transformation_multiplier(pde_data,
                                                                                                               leading_variable=1,
                                                                                                               derivation_order=2)
    pde_basis = deriv_1_pce - 4 * deriv_0_pce

    return pde_basis


# Definition of the source term (zero in this example)
def pde_res(s):
    return np.zeros(len(s))


# Definition of the function for sampling of boundary conditions
def bc_sampling(nsim=1000):
    # BC sampling

    nsim_half = round(nsim / 2)
    sample = np.zeros((nsim, 2))
    real_ogrid_1d = ortho_grid(nsim_half, 1, 0.0, 2.0)[:, 0]

    sample[:nsim_half, 0] = np.zeros(int(nsim / 2))
    sample[:nsim_half, 1] = real_ogrid_1d

    sample[nsim_half:, 0] = np.ones(nsim_half)
    sample[nsim_half:, 1] = real_ogrid_1d

    return sample


# Definition of the function for sampling and evaluation of BC. Function is utilized for estimation of an error once
# the PCE is created.
def bc_res(nsim, pce):
    bc_x = bc_sampling(nsim)
    bc_s = polynomial_chaos.Polynomials.standardize_sample(bc_x, pce.polynomial_basis.distributions)

    der_order = 0
    deriv_0_pce = np.sum(
        derivative_basis(bc_s, pce, derivative_order=der_order, leading_variable=0) * ((2 / 1) ** der_order) * np.array(
            pce.coefficients).T, axis=1)

    bc_init_x, bc_init_y = init_sampling(nsim)
    bc_init_s = polynomial_chaos.Polynomials.standardize_sample(bc_init_x, pce.polynomial_basis.distributions)

    der_order = 0
    deriv_0_init = np.sum(
        derivative_basis(bc_init_s, pce, derivative_order=der_order, leading_variable=0) * (
                    (2 / 1) ** der_order) * np.array(
            pce.coefficients).T, axis=1)

    der_order = 1
    deriv_1_init = np.sum(
        derivative_basis(bc_init_s, pce, derivative_order=der_order, leading_variable=1) * (
                    (2 / 1) ** der_order) * np.array(
            pce.coefficients).T, axis=1)

    return deriv_0_pce + np.abs(deriv_0_init - bc_init_y[:, 0]) + deriv_1_init


# Definition of the function for sampling of initial conditions
def init_sampling(nsim=1000):
    init_x = joint.rvs(nsim)

    init_x[:, 0] = ortho_grid(nsim, 1, 0.0, 1.0)[:, 0]

    # initial conditions are defined for t=0
    init_x[:, 1] = np.zeros(nsim)

    init_y = np.sin(np.pi * init_x[:, 0:1])
    return init_x, init_y


#  Definition of the function for sampling of virtual points (PDE is satisfied in virtual points)
def virt_sampling(nsim=1000):
    domain_samples = LatinHypercubeSampling(distributions=joint,
                                            criterion=Centered(),
                                            nsamples=nsim)._samples.reshape(-1, 2)

    return domain_samples


# Definition of the reference solution for an error estimation
def ref_sol(x):
    return np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:])


# %% md
#
# Wave equation is parametrized by two deterministic input variables (coordinate :math:`x` and time :math:`t`)
# modeled as Uniform random variables.

# number of input random variables
nvar = 2

#   definition of a joint probability distribution
dist1 = Uniform(loc=0, scale=1)
dist2 = Uniform(loc=0, scale=2)
marg = [dist1, dist2]
joint = JointIndependent(marginals=marg)

# %% md
#
# The physical domain is defined by two variables :math:`x \in [0,1], t \in [0,2]`


# define geometry of physical space

geometry_xmin = np.array([0, 0])
geometry_xmax = np.array([1, 2])

# %% md
#
# Boundary conditions are prescribed as
#
# .. math:: \frac{d u(x,0)}{d t}=0, \qquad u(0,t)=u(1,t) = 0, \qquad u(x,0) =\sin{(\pi x)}


# number of BC samples
nbc = 2 * 10
ninit = 10

# derivation orders of prescribed BCs
der_orders = [0, 1, 0]
# normals associated to prescribed BCs
bc_normals = np.array([0, 1, 1])
# sampling of BC points
bc_x = bc_sampling(nbc)
bc_y = np.zeros(len(bc_x))
bc_x_init, bc_y_init = init_sampling(ninit)

bc_coord = [bc_x, bc_x_init, bc_x_init]
bc_resy = [bc_y, np.zeros(len(bc_x_init)), bc_y_init[:, 0]]
# %% md
#
# Here we construct an object containing general physical information (geometry, boundary conditions)


pde_data = PdeData(geometry_xmax, geometry_xmin, der_orders, bc_normals, bc_coord, bc_resy)

# %% md
#
# Further we construct an object containing PDE physical data and PC :math:`^2` definitions of PDE


pde_pce = PdePCE(pde_data, pde_func, pde_source=pde_res, virtual_points_function=virt_sampling, boundary_condition_function=bc_sampling, boundary_conditions=bc_res)

# %% md
#
# Get dirichlet boundary conditions from PDE data object that are further used for construction of initial PCE object.


# extract dirichlet BC
dirichlet_bc = pde_data.dirichlet
x_train = dirichlet_bc[:, :-1]
y_train = dirichlet_bc[:, -1]

# %% md
#
# Finally, p-adaptive PC :math:`^2` is constructed using Karush-Kuhn-Tucker normal equations solved by ordinary least squares:


# construct PC^2
least_squares = LeastSquareRegression()
errors = []
best_err = 1000
for p in range(12, 15):
    PCEorder = p
    polynomial_basis = TotalDegreeBasis(joint, PCEorder, hyperbolic=1)

    #  create initial PCE object containing basis, regression method and dirichlet BC
    initpce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    initpce.set_data(x_train, y_train)

    # construct a PC^2 object combining pde_data, pde_pce and initial PCE objects
    pcpc = ConstrainedPCE(pde_data, pde_pce, initpce)

    # get coefficients of PC^2 by KKT-OLS
    pcpc.ols(n_error_points=10000)

    # get PC^2 error
    err = pcpc.ols_err

    errors.append(err)
    if err < best_err:
        best_p = p
        best_err = err

# construct the PC^2 with optimal polynomial order
polynomial_basis = TotalDegreeBasis(joint, best_p, hyperbolic=1)

#  create initial PCE object containing basis, regression method and dirichlet BC
initpce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
initpce.set_data(x_train, y_train)

# construct a PC^2 object combining pde_data, pde_pce and initial PCE objects
pcpc = ConstrainedPCE(pde_data, pde_pce, initpce)

# get coefficients of PC^2 by KKT-OLS
pcpc.ols(n_error_points=10000)
print('\np adaptive results: ')
print('Best polynomial order: ', best_p)
print('Sum of errors in PDE, BCs and training data: ', best_err)

# %% md
#
# PC :math:`^2` can be further used for a fast evaluation of the Wave equation. Obtained results are compared to
# an analytical reference solution in the following plot.


# plot the results and calculate absolute error
real_ogrid = ortho_grid(200, nvar, 0.0, 2.0)
real_ogrid[:, 0] = real_ogrid[:, 0] / 2

yy_val_pce = pcpc.initial_pce.predict(real_ogrid).flatten()
yy_val_true = ref_sol(real_ogrid).flatten()
abs_err = np.abs(yy_val_true - yy_val_pce)

print('\nComparison to the reference solution:')
print('Mean squared error of PC^2: ', np.mean(abs_err ** 2))

import matplotlib.pyplot as plt

vmin = yy_val_pce.min()
vmax = yy_val_pce.max()
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.jet

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
colors = cmap(norm(yy_val_pce))
ax[0].scatter(real_ogrid[:, 0], real_ogrid[:, 1], c=colors)

ax[0].scatter(pcpc.virtual_x[:, 0], pcpc.virtual_x[:, 1], c='green')
ax[0].scatter(bc_x_init[:, 0], bc_x_init[:, 1], c='black')
ax[0].scatter(x_train[:, 0], x_train[:, 1], c='black')

cmap_err = plt.cm.Reds
abs_err = np.abs(yy_val_true - yy_val_pce)
vmin_err = 0
vmax_err = abs_err.max()
norm_err = plt.Normalize(vmin=vmin_err, vmax=vmax_err)

colors = cmap_err(norm_err(abs_err))
ax[1].scatter(real_ogrid[:, 0], real_ogrid[:, 1], c=colors)

ax[0].set_title('PC$^2$ approximation')
ax[0].set_xlabel('coordinate $x$')
ax[0].set_ylabel('time $t$')

ax[1].set_title('Absolute error')
ax[1].set_xlabel('coordinate $x$')
fig.colorbar(plt.cm.ScalarMappable(norm=norm_err, cmap=cmap_err), ax=ax[1])
plt.show()
