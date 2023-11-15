import numpy as np
from UQpy.distributions.collection import Uniform, Normal
from scipy import special as sp
from scipy.special import legendre

def transformation_multiplier(original_geometry, var, derivation_order=1):
    """
    Get transformation multiplier for derivatives of PCE basis functions (assuming Uniform distribution)
    :param original_geometry: number of samples per dimension, i.e. total number is n**nvar
    :param var: number of dimensions
    :param derivation_order: order of derivative
    :return: multiplier reflecting a different sizes of physical and standardized spaces
    """

    size = np.abs(original_geometry.xmax[var] - original_geometry.xmin[var])
    multplier = (2 / size) ** derivation_order

    return multplier


def ortho_grid(n, nvar, xmin=-1, xmax=1):
    """
    Create orthogonal grid of samples.
    :param n: number of samples per dimension, i.e. total number is n**nvar
    :param nvar: number of dimensions
    :param xmin: lower bound of hypercube
    :param xmax: upper bound of hypercube
    :return: generated grid of samples
    """

    xrange = (xmax - xmin) / 2
    nsim = n ** nvar
    x = np.linspace(xmin + xrange / n, xmax - xrange / n, n)
    x_list = [x] * nvar
    X = np.meshgrid(*x_list)
    grid = np.array(X).reshape((nvar, nsim)).T
    return grid


def derivative_basis(s, pce, der_order=0, variable=None):
    """
    Create orthogonal grid of samples.
    :param s: samples in standardized space for an evaluation of derived basis
    :param pce: an object of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class
    :param der_order: order of derivative
    :param variable: leading variable of derivatives
    :return: evaluated derived basis
    """

    multindex = pce.multi_index_set
    joint_distribution = pce.polynomial_basis.distributions

    card_basis, nvar = multindex.shape

    if nvar == 1:
        marginals = [joint_distribution]
    else:
        marginals = joint_distribution.marginals

    mask_herm = [type(marg) == Normal for marg in marginals]
    mask_lege = [type(marg) == Uniform for marg in marginals]
    if variable is not None:

        ns = multindex[:, variable]
        polysd = []

        if mask_lege[variable]:

            for n in ns:
                polysd.append(legendre(n).deriv(der_order))

            prep_l_deriv = np.sqrt((2 * multindex[:, variable] + 1)).reshape(-1, 1)

            prep_deriv = []
            for poly in polysd:
                prep_deriv.append(np.polyval(poly, s[:, variable]).reshape(-1, 1))

            prep_deriv = np.array(prep_deriv)

        mask_herm[variable] = False
        mask_lege[variable] = False

    prep_hermite = sp.eval_hermitenorm(multindex[:, mask_herm][:, np.newaxis, :], s[:, mask_herm])
    prep_legendre = sp.eval_legendre(multindex[:, mask_lege][:, np.newaxis, :], s[:, mask_lege])

    prep_fact = np.sqrt(sp.factorial(multindex[:, mask_herm]))
    prep = np.sqrt((2 * multindex[:, mask_lege] + 1))

    multivariate_basis = np.prod(prep_hermite / prep_fact[:, np.newaxis, :], axis=2).T
    multivariate_basis *= np.prod(prep_legendre * prep[:, np.newaxis, :], axis=2).T

    if variable is not None:
        multivariate_basis *= np.prod(prep_deriv * prep_l_deriv[:, np.newaxis, :], axis=2).T

    return multivariate_basis


