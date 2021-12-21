# Test all distributions available in UQpy, using the cdf method or pdf method for multivariate distributions

from UQpy.distributions import *
import numpy as np


def test_beta():
    result = Beta(a=1., b=2.).cdf(x=0.8)
    assert result == 0.96


def test_cauchy():
    assert np.round(Cauchy().cdf(x=0.8), 3) == 0.715


def test_chi_square():
    assert np.round(ChiSquare(df=5.).cdf(x=0.8), 3) == 0.023


def test_exponential():
    assert np.round(Exponential().cdf(x=0.8), 3) == 0.551


def test_gamma():
    assert np.round(Gamma(a=2.).cdf(x=0.8), 3) == 0.191


def test_gen_extreme():
    assert GeneralizedExtreme(c=2.).cdf(x=0.8) == 1.


def test_inverse_gauss():
    assert np.round(InverseGauss(mu=2.).cdf(x=0.8), 3) == 0.411


def test_laplace():
    assert np.round(Laplace().cdf(x=0.8), 3) == 0.775


def test_levy():
    assert np.round(Levy().cdf(x=0.8), 3) == 0.264


def test_logistic():
    assert np.round(Logistic().cdf(x=0.8), 3) == 0.690


def test_lognormal():
    assert np.round(Lognormal(s=2.).cdf(x=0.8), 3) == 0.456


def test_maxwell():
    assert np.round(Maxwell().cdf(x=0.8), 3) == 0.113


def test_normal():
    assert np.round(Normal().cdf(x=0.8), 3) == 0.788


def test_pareto():
    assert np.round(Pareto(b=2.).cdf(x=1.1), 3) == 0.174


def test_poisson():
    assert np.round(Poisson(mu=2.).cdf(x=1.), 3) == 0.406


def test_rayleigh():
    assert np.round(Rayleigh().cdf(x=0.8), 3) == 0.274


def test_truncated_normal():
    assert np.round(TruncatedNormal(a=-1., b=1.).cdf(x=0.8), 3) == 0.922


# For multinomial, mvnormal, more tests are needed


def test_multinomial_1():
    assert Multinomial(n=5, p=[0.2, 0.3, 0.5]).pmf(x=[1, 1, 3]) == 0.15


def test_multinomial_2():
    assert np.round(Multinomial(n=5, p=[0.2, 0.3, 0.5]).log_pmf(x=[1, 1, 3]), 3) == -1.897


def test_multinomial_3():
    samples = Multinomial(n=5, p=[0.2, 0.3, 0.5]).rvs(nsamples=2, random_state=123)
    assert np.all(samples == np.array([[1, 1, 3], [0, 2, 3]]))


def test_multinomial_4():
    multinomial = Multinomial(n=5, p=[0.2, 0.3, 0.5])
    moments = multinomial.moments(moments2return='m')
    assert np.all(moments == [1., 1.5, 2.5])


def test_multinomial_5():
    cov = Multinomial(n=5, p=[0.2, 0.3, 0.5]).moments(moments2return='v')
    assert np.all(np.round(cov, 2) == np.array([[0.80, -0.30, -0.50], [-0.30, 1.05, -0.75], [-0.50, -0.75, 1.25]]))


def test_multinomial_6():
    moments = Multinomial(n=5, p=[0.2, 0.3, 0.5]).moments(moments2return='mv')
    true_values = (np.array([1., 1.5, 2.5]),
                   np.array([[0.80, -0.30, -0.50], [-0.30, 1.05, -0.75], [-0.50, -0.75, 1.25]]))
    assert np.all(moments[0] == true_values[0]) and np.all(np.round(moments[1], 2) == true_values[1])


def test_mvnormal_1():
    assert np.round(MultivariateNormal(mean=[1., 2.], cov=3.).cdf(x=[0.8, 0.8]), 3) == 0.111


def test_mvnormal_2():
    assert np.round(MultivariateNormal(mean=[1., 2.], cov=3.).pdf(x=[0.8, 0.8]), 3) == 0.041


def test_mvnormal_3():
    assert np.round(MultivariateNormal(mean=[1., 2.], cov=3.).log_pdf(x=[0.8, 0.8]), 3) == -3.183


def test_mvnormal_4():
    data = np.array([[0., 0.9], [0.1, 1.], [-0.1, 1.1]])
    true_mean = np.array([0., 1.])
    true_cov = np.array([[0.010, -0.005], [-0.005, 0.010]])
    dict_fit = MultivariateNormal(mean=None, cov=None).fit(data=data)
    assert np.all(dict_fit['mean'] == true_mean) and np.all(np.round(dict_fit['cov'], 3) == true_cov)


def test_mvnormal_5():
    samples = MultivariateNormal(mean=[1., 2.], cov=1.).rvs(nsamples=3, random_state=123)
    assert np.all(np.round(samples, 3) == np.array([[-0.086, 2.997], [1.283, 0.494], [0.421, 3.651]]))


def test_mvnormal_6():
    assert np.all(MultivariateNormal(mean=[1., 2.], cov=3.).moments(moments2return='m') == [1., 2.])


def test_mvnormal_7():
    assert np.all(MultivariateNormal(mean=[1., 2.], cov=3.).moments(moments2return='v') == 3.)


def test_mvnormal_8():
    moments = MultivariateNormal(mean=[1., 2.], cov=3.).moments(moments2return='mv')
    assert np.all(moments[0] == [1., 2.]) and moments[1] == 3.


# Check copulas
unif = np.array([0.4, 0.9]).reshape((1, 2))


def test_clayton():
    assert np.round(Clayton(theta=2.).evaluate_cdf(unit_uniform_samples=unif), 3) == 0.393


def test_frank():
    assert np.round(Frank(theta=2.).evaluate_cdf(unit_uniform_samples=unif), 3) == 0.379


def test_gumbel_1():
    assert np.round(Gumbel(theta=2.).evaluate_cdf(unit_uniform_samples=unif), 3) == 0.398


def test_gumbel_2():
    assert np.round(Gumbel(theta=2.).evaluate_pdf(unit_uniform_samples=unif), 3) == 0.261


# Check JointInd and JointCopula

marginals = [Normal(loc=2., scale=2.), Lognormal(s=1., loc=0., scale=np.exp(1))]
dist_joint = JointIndependent(marginals=marginals)
dist_joint_copula = JointCopula(marginals=marginals, copula=Gumbel(theta=2.))


def test_joint_ind_1():
    marginals_ = [Normal(loc=2., scale=2.), Lognormal(s=1., loc=0., scale=np.exp(1))]
    dist_joint_ = JointIndependent(marginals=marginals_)
    dist_joint_.update_parameters(loc_0=3.)
    assert dist_joint_.get_parameters()['loc_0'] == 3.


def test_joint_ind_2():
    samples = dist_joint.rvs(nsamples=1, random_state=123)
    assert np.all(np.round(samples, 3) == [[-0.171, 0.918]])


def test_joint_ind_3():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    assert np.round(dist_joint.pdf(x=x), 3) == 0.029


def test_joint_ind_4():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    assert np.round(dist_joint.log_pdf(x=x), 3) == -3.553


def test_joint_ind_5():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    assert np.round(dist_joint.cdf(x=x), 3) == 0.010


def test_joint_ind_6():
    assert np.all(np.round(dist_joint.moments(moments2return='m'), 3) == [2., 4.482])


def test_joint_ind_7():
    marginals_ = [Normal(loc=None, scale=2.), Lognormal(s=1., loc=0., scale=np.exp(1))]
    dist_joint_ = JointIndependent(marginals=marginals_)
    data = np.array([[-0.17126121, 0.91793325], [3.99469089, 7.36946747], [2.565957, 3.60736828]])
    mle_fit = dist_joint_.fit(data=data)
    assert np.round(mle_fit['loc_0'], 3) == 2.130


def test_joint_copula_1():
    marginals_ = [Normal(loc=2., scale=2.), Lognormal(s=1., loc=0., scale=np.exp(1))]
    dist_joint_ = JointCopula(marginals=marginals_, copula=Gumbel(theta=3.))
    dist_joint_.update_parameters(theta_c=2.)
    assert dist_joint_.get_parameters()['theta_c'] == 2.


def test_joint_copula_3():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    assert np.round(dist_joint_copula.pdf(x=x), 3) == 0.045


def test_joint_copula_4():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    assert np.round(dist_joint_copula.log_pdf(x=x), 3) == -3.092


def test_joint_copula_5():
    x = np.array([0.5, 0.5]).reshape((1, 2))
    assert np.round(dist_joint_copula.cdf(x=x), 3) == 0.032
