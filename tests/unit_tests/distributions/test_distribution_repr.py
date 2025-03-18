from UQpy import distributions


def test_beta_repr():
    beta = distributions.Beta(1, 2, 3, 4)
    assert beta.__repr__() == "Beta(1, 2, loc=3, scale=4)"


def test_binomial_repr():
    binomial = distributions.Binomial(1, 2, 3)
    assert binomial.__repr__() == "Binomial(1, 2, loc=3)"


def test_cauchy_repr():
    cauchy = distributions.Cauchy(1, 2)
    assert cauchy.__repr__() == "Cauchy(loc=1, scale=2)"


def test_chi_square_repr():
    chi_square = distributions.ChiSquare(1, 2, 3)
    assert chi_square.__repr__() == "ChiSquare(1, loc=2, scale=3)"


def test_exponential_repr():
    exponential = distributions.Exponential(1, 2)
    assert exponential.__repr__() == "Exponential(loc=1, scale=2)"


def test_gamma_repr():
    gamma = distributions.Gamma(1, 2, 3)
    assert gamma.__repr__() == "Gamma(1, loc=2, scale=3)"


def test_generalized_extreme_repr():
    generalized_extreme = distributions.GeneralizedExtreme(1, 2, 3)
    assert generalized_extreme.__repr__() == "GeneralizedExtreme(1, loc=2, scale=3)"


def test_inverse_gaussian_repr():
    inverse_gauss = distributions.InverseGauss(1, 2, 3)
    assert inverse_gauss.__repr__() == "InverseGauss(1, loc=2, scale=3)"


def test_joint_copula_repr():
    joint_copula = distributions.JointCopula(
        [distributions.Normal(loc=2.0), distributions.Uniform()],
        copula=distributions.copulas.Frank(1.0),
    )
    assert (
        joint_copula.__repr__()
        == "JointCopula([Normal(loc=2.0), Uniform()], Frank(1.0))"
    )


def test_joint_independent_repr():
    joint_independent = distributions.JointIndependent(
        [distributions.Beta(1, 2), distributions.Cauchy()]
    )
    assert joint_independent.__repr__() == "JointIndependent([Beta(1, 2), Cauchy()])"


def test_laplace_repr():
    laplace = distributions.Laplace(1, 2)
    assert laplace.__repr__() == "Laplace(loc=1, scale=2)"


def test_levy_repr():
    levy = distributions.Levy(1, 2)
    assert levy.__repr__() == "Levy(loc=1, scale=2)"


def test_logistic_repr():
    logistic = distributions.Logistic(1, 2)
    assert logistic.__repr__() == "Logistic(loc=1, scale=2)"


def test_lognormal_repr():
    lognormal = distributions.Lognormal(1, 2, 3)
    assert lognormal.__repr__() == "Lognormal(1, loc=2, scale=3)"


def test_maxwell_repr():
    maxwell = distributions.Maxwell(1, 2)
    assert maxwell.__repr__() == "Maxwell(loc=1, scale=2)"


def test_multinomial_repr():
    multinomial = distributions.Multinomial(1, [2.0, 3.0])
    assert multinomial.__repr__() == "Multinomial(1, [2.0, 3.0])"


def test_multivariate_normal_repr():
    multivariate_normal = distributions.MultivariateNormal(
        [1, 2], [[1.0, 0.4], [0.4, 1.0]]
    )
    assert (
        multivariate_normal.__repr__()
        == "MultivariateNormal([1, 2], cov=[[1.0, 0.4], [0.4, 1.0]])"
    )


def test_normal_repr():
    normal = distributions.Normal(1, 2)
    assert normal.__repr__() == "Normal(loc=1, scale=2)"


def test_pareto_repr():
    pareto = distributions.Pareto(1, 2, 3)
    assert pareto.__repr__() == "Pareto(1, loc=2, scale=3)"


def test_poisson_repr():
    poisson = distributions.Poisson(1, 2)
    assert poisson.__repr__() == "Poisson(1, loc=2)"


def test_rayleigh_repr():
    rayleigh = distributions.Rayleigh(1, 2)
    assert rayleigh.__repr__() == "Rayleigh(loc=1, scale=2)"


def test_truncated_normal_repr():
    truncated_normal = distributions.TruncatedNormal(1, 2, 3, 4)
    assert truncated_normal.__repr__() == "TruncatedNormal(1, 2, loc=3, scale=4)"


def test_uniform_repr():
    uniform = distributions.Uniform(1, 2)
    assert uniform.__repr__() == "Uniform(loc=1, scale=2)"
