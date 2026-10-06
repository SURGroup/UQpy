# Test all distributions available in UQpy, using the cdf method or pdf method for multivariate distributions

from UQpy.distributions import *
import numpy as np


def test_beta():
    result = Beta(a=1.0, b=2.0).cdf(x=0.8)
    np.testing.assert_allclose(result, 0.96, atol=1e-3)


def test_cauchy():
    np.testing.assert_allclose(Cauchy().cdf(x=0.8), 0.715, atol=1e-3)


def test_chi_square():
    np.testing.assert_allclose(ChiSquare(df=5.0).cdf(x=0.8), 0.023, atol=1e-3)


def test_exponential():
    np.testing.assert_allclose(Exponential().cdf(x=0.8), 0.551, atol=1e-3)


def test_gamma():
    np.testing.assert_allclose(Gamma(a=2.0).cdf(x=0.8), 0.191, atol=1e-3)


def test_gen_extreme():
    np.testing.assert_allclose(GeneralizedExtreme(c=2.0).cdf(x=0.8), 1.0, atol=1e-3)


def test_inverse_gauss():
    np.testing.assert_allclose(InverseGauss(mu=2.0).cdf(x=0.8), 0.411, atol=1e-3)


def test_laplace():
    np.testing.assert_allclose(Laplace().cdf(x=0.8), 0.775, atol=1e-3)


def test_levy():
    np.testing.assert_allclose(Levy().cdf(x=0.8), 0.264, atol=1e-3)


def test_logistic():
    np.testing.assert_allclose(Logistic().cdf(x=0.8), 0.690, atol=1e-3)


def test_lognormal():
    np.testing.assert_allclose(Lognormal(s=2.0).cdf(x=0.8), 0.456, atol=1e-3)


def test_maxwell():
    np.testing.assert_allclose(Maxwell().cdf(x=0.8), 0.113, atol=1e-3)


def test_normal():
    np.testing.assert_allclose(Normal().cdf(x=0.8), 0.788, atol=1e-3)


def test_pareto():
    np.testing.assert_allclose(Pareto(b=2.0).cdf(x=1.1), 0.174, atol=1e-3)


def test_poisson():
    np.testing.assert_allclose(Poisson(mu=2.0).cdf(x=1.0), 0.406, atol=1e-3)


def test_rayleigh():
    np.testing.assert_allclose(Rayleigh().cdf(x=0.8), 0.274, atol=1e-3)


def test_truncated_normal():
    np.testing.assert_allclose(
        TruncatedNormal(a=-1.0, b=1.0).cdf(x=0.8), 0.922, atol=1e-3
    )


# For multinomial, mvnormal, more tests are needed


def test_multinomial_1():
    np.testing.assert_allclose(
        Multinomial(n=5, p=[0.2, 0.3, 0.5]).pmf(x=[1, 1, 3]), 0.15
    )


def test_multinomial_2():
    np.testing.assert_allclose(
        Multinomial(n=5, p=[0.2, 0.3, 0.5]).log_pmf(x=[1, 1, 3]), -1.897, atol=1e-3
    )


def test_multinomial_3():
    samples = Multinomial(n=5, p=[0.2, 0.3, 0.5]).rvs(nsamples=2, random_state=123)
    np.testing.assert_allclose(samples, np.array([[1, 1, 3], [0, 2, 3]]))


def test_multinomial_4():
    multinomial = Multinomial(n=5, p=[0.2, 0.3, 0.5])
    moments = multinomial.moments(moments2return="m")
    np.testing.assert_allclose(moments, [1.0, 1.5, 2.5])


def test_multinomial_5():
    cov = Multinomial(n=5, p=[0.2, 0.3, 0.5]).moments(moments2return="v")
    np.testing.assert_allclose(
        cov,
        [[0.80, -0.30, -0.50], [-0.30, 1.05, -0.75], [-0.50, -0.75, 1.25]],
        atol=1e-2,
    )


def test_multinomial_6():
    moments = Multinomial(n=5, p=[0.2, 0.3, 0.5]).moments(moments2return="mv")
    true_values = (
        np.array([1.0, 1.5, 2.5]),
        np.array([[0.80, -0.30, -0.50], [-0.30, 1.05, -0.75], [-0.50, -0.75, 1.25]]),
    )
    np.testing.assert_allclose(moments[0], true_values[0])
    np.testing.assert_allclose(moments[1], true_values[1], atol=1e-2)


def test_mvnormal_1():
    np.testing.assert_allclose(
        MultivariateNormal(mean=[1.0, 2.0], cov=3.0).cdf(x=[0.8, 0.8]), 0.111, atol=1e-3
    )


def test_mvnormal_2():
    np.testing.assert_allclose(
        MultivariateNormal(mean=[1.0, 2.0], cov=3.0).pdf(x=[0.8, 0.8]), 0.041, atol=1e-3
    )


def test_mvnormal_3():
    np.testing.assert_allclose(
        MultivariateNormal(mean=[1.0, 2.0], cov=3.0).log_pdf(x=[0.8, 0.8]),
        -3.183,
        atol=1e-3,
    )


def test_mvnormal_4():
    data = np.array([[0.0, 0.9], [0.1, 1.0], [-0.1, 1.1]])
    true_mean = np.array([0.0, 1.0])
    true_cov = np.array([[0.010, -0.005], [-0.005, 0.010]])
    dict_fit = MultivariateNormal(mean=None, cov=None).fit(data=data)
    np.testing.assert_allclose(dict_fit["mean"], true_mean)
    np.testing.assert_allclose(dict_fit["cov"], true_cov, atol=1e-3)


def test_mvnormal_5():
    samples = MultivariateNormal(mean=[1.0, 2.0], cov=1.0).rvs(
        nsamples=3, random_state=123
    )
    np.testing.assert_allclose(
        samples, np.array([[-0.086, 2.997], [1.283, 0.494], [0.421, 3.651]]), atol=1e-3
    )


def test_mvnormal_6():
    np.testing.assert_allclose(
        MultivariateNormal(mean=[1.0, 2.0], cov=3.0).moments(moments2return="m"),
        [1.0, 2.0],
    )


def test_mvnormal_7():
    np.testing.assert_allclose(
        MultivariateNormal(mean=[1.0, 2.0], cov=3.0).moments(moments2return="v"), 3.0
    )


def test_mvnormal_8():
    moments = MultivariateNormal(mean=[1.0, 2.0], cov=3.0).moments(moments2return="mv")
    np.testing.assert_allclose(moments[0], [1.0, 2.0])
    np.testing.assert_allclose(moments[1], 3.0)


# Check copulas
unif = np.array([0.4, 0.9]).reshape((1, 2))


def test_clayton():
    np.testing.assert_allclose(
        Clayton(theta=2.0).evaluate_cdf(unit_uniform_samples=unif), 0.393, atol=1e-3
    )


def test_frank():
    np.testing.assert_allclose(
        Frank(theta=2.0).evaluate_cdf(unit_uniform_samples=unif), 0.379, atol=1e-3
    )


def test_gumbel_1():
    np.testing.assert_allclose(
        Gumbel(theta=2.0).evaluate_cdf(unit_uniform_samples=unif), 0.398, atol=1e-3
    )


def test_gumbel_2():
    np.testing.assert_allclose(
        Gumbel(theta=2.0).evaluate_pdf(unit_uniform_samples=unif), 0.261, atol=1e-3
    )


# Check JointInd and JointCopula

marginals = [Normal(loc=2.0, scale=2.0), Lognormal(s=1.0, loc=0.0, scale=np.exp(1))]
dist_joint = JointIndependent(marginals=marginals)
dist_joint_copula = JointCopula(marginals=marginals, copula=Gumbel(theta=2.0))


def test_joint_ind_1():
    marginals_ = [
        Normal(loc=2.0, scale=2.0),
        Lognormal(s=1.0, loc=0.0, scale=np.exp(1)),
    ]
    dist_joint_ = JointIndependent(marginals=marginals_)
    dist_joint_.update_parameters(loc_0=3.0)
    np.testing.assert_allclose(dist_joint_.get_parameters()["loc_0"], 3.0)


def test_joint_ind_2():
    samples = dist_joint.rvs(nsamples=1, random_state=123)
    np.testing.assert_allclose(samples, np.array([[-0.171, 0.918]]), atol=1e-3)


def test_joint_ind_3():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    np.testing.assert_allclose(dist_joint.pdf(x=x), 0.029, atol=1e-3)


def test_joint_ind_4():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    np.testing.assert_allclose(dist_joint.log_pdf(x=x), -3.553, atol=1e-3)


def test_joint_ind_5():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    np.testing.assert_allclose(dist_joint.cdf(x=x), 0.010, atol=1e-3)


def test_joint_ind_6():
    np.testing.assert_allclose(
        dist_joint.moments(moments2return="m"), [2.0, 4.482], atol=1e-3
    )


def test_joint_ind_7():
    marginals_ = [
        Normal(loc=None, scale=2.0),
        Lognormal(s=1.0, loc=0.0, scale=np.exp(1)),
    ]
    dist_joint_ = JointIndependent(marginals=marginals_)
    data = np.array(
        [[-0.17126121, 0.91793325], [3.99469089, 7.36946747], [2.565957, 3.60736828]]
    )
    mle_fit = dist_joint_.fit(data=data)
    np.testing.assert_allclose(mle_fit["loc_0"], 2.130, atol=1e-3)


def test_joint_copula_1():
    marginals_ = [
        Normal(loc=2.0, scale=2.0),
        Lognormal(s=1.0, loc=0.0, scale=np.exp(1)),
    ]
    dist_joint_ = JointCopula(marginals=marginals_, copula=Gumbel(theta=3.0))
    dist_joint_.update_parameters(theta_c=2.0)
    np.testing.assert_allclose(dist_joint_.get_parameters()["theta_c"], 2.0)


def test_joint_copula_3():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    np.testing.assert_allclose(dist_joint_copula.pdf(x=x), 0.045, atol=1e-3)


def test_joint_copula_4():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    np.testing.assert_allclose(dist_joint_copula.log_pdf(x=x), -3.092, atol=1e-3)


def test_joint_copula_5():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    np.testing.assert_allclose(dist_joint_copula.cdf(x=x), 0.032, atol=1e-3)
