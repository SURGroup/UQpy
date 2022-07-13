import pytest

from UQpy.distributions import JointIndependent, Normal
from UQpy.sampling import MonteCarloSampling
from UQpy.distributions import Uniform
from UQpy.sensitivity.PceSensitivity import PceSensitivity
from UQpy.surrogates import *
import numpy as np

from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import TotalDegreeBasis
from UQpy.surrogates.polynomial_chaos.polynomials.TensorProductBasis import TensorProductBasis

np.random.seed(1)
max_degree, n_samples = 2, 10
dist = Uniform(loc=0, scale=10)


def func(x):
    return x * np.sin(x) / 10


x = dist.rvs(n_samples)
x_test = dist.rvs(n_samples)
y = func(x)


# Unit tests
def test_1():
    """
    Test td basis
    """
    polynomials = TotalDegreeBasis(dist, max_degree).polynomials
    value = polynomials[1].evaluate(x)[0]
    assert round(value, 4) == -0.2874


def test_2():
    """
    Test tp basis
    """
    polynomial_basis = TensorProductBasis(distributions=dist,
                                                                   max_degree=max_degree).polynomials
    value = polynomial_basis[1].evaluate(x)[0]
    assert round(value, 4) == -0.2874


def test_3():
    """
    Test PCE coefficients w/ lasso
    """
    polynomial_basis = TensorProductBasis(dist, max_degree)
    lasso = LassoRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=lasso)
    pce.fit(x, y)
    assert round(pce.coefficients[0][0], 4) == 0.0004


#
def test_4():
    """
    Test PCE coefficients w/ ridge
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    ridge = RidgeRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=ridge)
    pce.fit(x, y)
    assert round(pce.coefficients[0][0], 4) == 0.0276


#
def test_5():
    """
    Test PCE coefficients w/ lstsq
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    assert round(pce.coefficients[0][0], 4) == 0.2175


#
def test_6():
    """
    Test PCE prediction
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    y_test = pce.predict(x_test)
    assert round(y_test[0][0], 4) == -0.1607


#
def test_7():
    """
    Test Sobol indices
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    pce_sensitivity = PceSensitivity(pce)
    first_order_sobol = pce_sensitivity.calculate_first_order_indices()
    assert round(first_order_sobol[0][0], 3) == 1.0


#
def test_8():
    """
    Test Sobol indices
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    pce_sensitivity = PceSensitivity(pce)

    total_order_sobol = pce_sensitivity.calculate_total_order_indices()
    assert round(total_order_sobol[0][0], 3) == 1.0


#
def test_9():
    """
    Test Sobol indices
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    pce_sensitivity = PceSensitivity(pce)


#
def test_10():
    """
    Test Sobol indices
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    pce_sensitivity = PceSensitivity(pce)
    with pytest.raises(ValueError):
        generalized_total_order_sobol = pce_sensitivity.calculate_generalized_total_order_indices()


#
def test_11():
    """
    PCE mean
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    mean, _ = pce.get_moments()
    assert round(mean, 3) == 0.218


#
def test_12():
    """
    PCE variance
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, y)
    _, variance = pce.get_moments()
    assert round(variance, 3) == 0.185


def function(x):
    # without square root
    u1 = x[:, 4] * np.cos(x[:, 0])
    u2 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1))
    u3 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.cos(
        np.sum(x[:, :3], axis=1))
    u4 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.cos(
        np.sum(x[:, :3], axis=1)) + x[:, 7] * np.cos(np.sum(x[:, :4], axis=1))

    v1 = x[:, 4] * np.sin(x[:, 0])
    v2 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1))
    v3 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.sin(
        np.sum(x[:, :3], axis=1))
    v4 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.sin(
        np.sum(x[:, :3], axis=1)) + x[:, 7] * np.sin(np.sum(x[:, :4], axis=1))

    return (u1 + u2 + u3 + u4) ** 2 + (v1 + v2 + v3 + v4) ** 2


dist_1 = Uniform(loc=0, scale=2 * np.pi)
dist_2 = Uniform(loc=0, scale=1)

marg = [dist_1] * 4
marg_1 = [dist_2] * 4
marg.extend(marg_1)

joint = JointIndependent(marginals=marg)

n_samples_2 = 10
mcs_2 = MonteCarloSampling(distributions=joint, nsamples=n_samples_2, random_state=0)
x_2 = mcs_2.samples
y_2 = function(x_2)

polynomial_basis = TotalDegreeBasis(joint, 2)
least_squares = LeastSquareRegression()
pce_2 = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
pce_2.fit(x_2, y_2)


def test_17():
    """
    Test Sobol indices for vector-valued quantity of interest on the random inputs
    """
    pce_sensitivity = PceSensitivity(pce_2)
    pce_sensitivity.run()
    generalized_first_sobol = pce_sensitivity.generalized_first_order_indices
    assert round(generalized_first_sobol[0], 4) == 0.0137


def test_lotka_volterra_generalized_sobol():
    import numpy as np
    import math
    from scipy import integrate
    from UQpy.distributions import Uniform, JointIndependent
    from UQpy.surrogates.polynomial_chaos import TotalDegreeBasis, \
        LeastSquareRegression, \
        PolynomialChaosExpansion
    from UQpy.sensitivity.PceSensitivity import PceSensitivity

    ### function to be approximated
    def LV(a, b, c, d, t):

        # X_f0 = np.array([     0. ,  0.])
        X_f1 = np.array([c / (d * b), a / b])

        def dX_dt(X, t=0):
            """ Return the growth rate of fox and rabbit populations. """
            return np.array([a * X[0] - b * X[0] * X[1],
                             -c * X[1] + d * b * X[0] * X[1]])

        X0 = np.array([10, 5])  # initials conditions: 10 rabbits and 5 foxes

        X, infodict = integrate.odeint(dX_dt, X0, t, full_output=True)

        return X, X_f1

    # set random seed for reproducibility
    np.random.seed(1)

    ### simulation parameters
    n = 512
    t = np.linspace(0, 25, n)

    ### Probability distributions of input parameters
    pdf1 = Uniform(loc=0.9, scale=0.1)  # a
    pdf2 = Uniform(loc=0.1, scale=0.05)  # b
    # pdf2 = Uniform(loc=8, scale=10)  # c
    # pdf2 = Uniform(loc=8, scale=10)  # d
    c = 1.5
    d = 0.75
    margs = [pdf1, pdf2]
    joint = JointIndependent(marginals=margs)

    print('Total degree: ', max_degree)
    polynomial_basis = TotalDegreeBasis(joint, max_degree)

    print('Size of basis:', polynomial_basis.polynomials_number)
    # training data
    sampling_coeff = 5
    print('Sampling coefficient: ', sampling_coeff)
    np.random.seed(42)
    n_samples = math.ceil(sampling_coeff * polynomial_basis.polynomials_number)
    print('Training data: ', n_samples)
    x_train = joint.rvs(n_samples)
    y_train = []
    for i in range(x_train.shape[0]):
        out, X_f1 = LV(x_train[i, 0], x_train[i, 1], c, d, t)
        y_train.append(out.flatten())
    print('Training sample size:', n_samples)

    # fit model
    least_squares = LeastSquareRegression()
    pce_metamodel = PolynomialChaosExpansion(polynomial_basis=polynomial_basis,
                                             regression_method=least_squares)
    pce_metamodel.fit(x_train, y_train)

    # approximation errors
    np.random.seed(43)
    n_samples_test = 5000
    x_test = joint.rvs(n_samples_test)
    y_test = []
    for i in range(x_test.shape[0]):
        out, X_f1 = LV(x_test[i, 0], x_test[i, 1], c, d, t)
        y_test.append(out.flatten())
    print('Test sample size:', n_samples_test)

    y_test_pce = pce_metamodel.predict(x_test)
    errors = np.abs(y_test_pce - y_test)
    l2_rel_err = np.linalg.norm(errors, axis=1) / np.linalg.norm(y_test, axis=1)

    l2_rel_err_mean = np.mean(l2_rel_err)
    print('Mean L2 relative error:', l2_rel_err_mean)

    # Sobol sensitivity analysis
    pce_sa = PceSensitivity(pce_metamodel)
    GS1 = pce_sa.calculate_generalized_first_order_indices()
    assert round(GS1[0], 4) == 0.2148
    assert round(GS1[1], 4) == 0.7426
    GST = pce_sa.calculate_generalized_total_order_indices()
    assert round(GST[0], 4) == 0.2574
    assert round(GST[1], 4) == 0.7852


def test_18():
    """
    Test Sobol indices for vector-valued quantity of interest on the random inputs
    """
    pce_sensitivity = PceSensitivity(pce_2)
    pce_sensitivity.run()
    generalized_total_sobol = pce_sensitivity.generalized_total_order_indices
    assert round(generalized_total_sobol[0], 4) == 0.4281


def test_19():
    """
    Test Higher statistical moments on Uniform distribution (skewness=0, kurtosis=1.8 from definition)
    """
    polynomial_basis = TotalDegreeBasis(dist, max_degree)
    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pce.fit(x, x)
    mean, var, skew, kurtosis = pce.get_moments(True)

    assert round(kurtosis, 3) == 1.8 and round(skew, 3) == 0


def test_20():
    """
    Test Higher statistical moments on Gaussian distribution (skewness=0, kurtosis=3 from definition)
    """
    dist_Gauss = Normal(loc=0, scale=1)

    mcs = MonteCarloSampling(dist, nsamples=n_samples, random_state=1)

    polynomial_basis = TotalDegreeBasis(dist_Gauss, max_degree)
    least_squares = LeastSquareRegression()
    pceGauss = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
    pceGauss.fit(x, x)
    mean, var, skew, kurtosis = pceGauss.get_moments(True)

    assert round(kurtosis, 3) == 3 and round(skew, 3) == 0


def functionLAR(x):
    u1 = x[:, 0] ** 2 + x[:, 2] ** 2 + x[:, 3] ** 2

    return u1


n_samples_2 = 100
x_2 = joint.rvs(n_samples_2)
y_2 = functionLAR(x_2)

polynomial_basis = TotalDegreeBasis(joint, 6)
least_squares = LeastSquareRegression()
pce_2 = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)
pce_2.fit(x_2, y_2)


def test_21():
    """
    Test Model Selection Algorithm based on Least Angle Regression (select only important basis functions from large set)
    """

    pce_lar = polynomial_chaos.regressions.LeastAngleRegression.model_selection(pce_2)
    pce2_lar_sens = PceSensitivity(pce_lar)

    assert all((np.argwhere(np.round(pce2_lar_sens.calculate_generalized_total_order_indices(), 3) > 0)
                == [[0], [2], [3]]))
