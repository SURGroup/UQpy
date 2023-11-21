import numpy as np
from scipy.special import legendre
from beartype import beartype
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.distributions.baseclass.Distribution import Distribution
from UQpy.distributions.collection.Normal import Normal
from UQpy.distributions.collection.Uniform import Uniform
from UQpy.surrogates import *
from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData
from scipy import special as sp


@beartype
def transformation_multiplier(data_object: PdeData, leading_variable, derivation_order=1):
    """
    Get transformation multiplier for derivatives of PCE basis functions (assuming Uniform distribution)
    :param data_object: :py:meth:`UQpy` :class:`PdeData` class containing geometry of physical space
    :param leading_variable: leading variable for derivation
    :param derivation_order: order of derivative
    :return: multiplier reflecting a different sizes of physical and standardized spaces
    """

    size = np.abs(data_object.xmax[leading_variable] - data_object.xmin[leading_variable])
    multiplier = (2 / size) ** derivation_order

    return multiplier


@beartype
def ortho_grid(n_samples: int, nvar: int, x_min: float = -1, x_max: float = 1):
    """
    Create orthogonal grid of samples.
    :param n_samples: number of samples per dimension, i.e. total number is n**nvar
    :param nvar: number of dimensions
    :param x_min: lower bound of hypercube
    :param x_max: upper bound of hypercube
    :return: generated grid of samples
    """

    xrange = (x_max - x_min) / 2
    nsim = n_samples ** nvar
    x = np.linspace(x_min + xrange / n_samples, x_max - xrange / n_samples, n_samples)
    x_list = [x] * nvar
    X = np.meshgrid(*x_list)
    grid = np.array(X).reshape((nvar, nsim)).T
    return grid


@beartype
def derivative_basis(standardized_sample: np.ndarray, pce: PolynomialChaosExpansion, derivative_order: int,
                     leading_variable: int):
    """
    Evaluate derivative basis of given pce object.
    :param standardized_sample: samples in standardized space for an evaluation of derived basis
    :param pce: an object of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class
    :param derivative_order: order of derivative
    :param leading_variable: leading variable of derivatives
    :return: evaluated derived basis
    """

    if derivative_order >= 0:
        multindex = pce.multi_index_set
        joint_distribution = pce.polynomial_basis.distributions

        multivariate_basis = construct_basis(standardized_sample, multindex, joint_distribution, derivative_order,
                                             leading_variable)
    else:
        raise Exception('derivative_basis function is defined only for positive derivative_order!')

    return multivariate_basis


@beartype
def construct_basis(standardized_sample: np.ndarray, multindex: np.ndarray,
                    joint_distribution: Distribution,
                    derivative_order: int = 0, leading_variable: int = 0):
    """
        Construct and evaluate derivative basis.
        :param standardized_sample: samples in standardized space for an evaluation of derived basis
        :param multindex: set of multi-indices corresponding to polynomial orders in basis set
        :param joint_distribution: joint probability distribution of input variables,
        an object of the :py:meth:`UQpy` :class:`Distribution` class
        :param derivative_order: order of derivative
        :param leading_variable: leading variable of derivatives
        :return: evaluated derived basis
        """

    card_basis, nvar = multindex.shape

    if nvar == 1:
        marginals = [joint_distribution]
    else:
        marginals = joint_distribution.marginals

    mask_herm = [type(marg) == Normal for marg in marginals]
    mask_lege = [type(marg) == Uniform for marg in marginals]
    if derivative_order >= 0:

        ns = multindex[:, leading_variable]
        polysd = []

        if mask_lege[leading_variable]:

            for n in ns:
                polysd.append(legendre(n).deriv(derivative_order))

            prep_l_deriv = np.sqrt((2 * multindex[:, leading_variable] + 1)).reshape(-1, 1)

            prep_deriv = []
            for poly in polysd:
                prep_deriv.append(np.polyval(poly, standardized_sample[:, leading_variable]).reshape(-1, 1))

            prep_deriv = np.array(prep_deriv)

        mask_herm[leading_variable] = False
        mask_lege[leading_variable] = False

    prep_hermite = sp.eval_hermitenorm(multindex[:, mask_herm][:, np.newaxis, :], standardized_sample[:, mask_herm])
    prep_legendre = sp.eval_legendre(multindex[:, mask_lege][:, np.newaxis, :], standardized_sample[:, mask_lege])

    prep_fact = np.sqrt(sp.factorial(multindex[:, mask_herm]))
    prep = np.sqrt((2 * multindex[:, mask_lege] + 1))

    multivariate_basis = np.prod(prep_hermite / prep_fact[:, np.newaxis, :], axis=2).T
    multivariate_basis *= np.prod(prep_legendre * prep[:, np.newaxis, :], axis=2).T

    if leading_variable is not None:
        multivariate_basis *= np.prod(prep_deriv * prep_l_deriv[:, np.newaxis, :], axis=2).T
    return multivariate_basis
