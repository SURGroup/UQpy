import numpy as np
from UQpy.Distributions import Distribution

p = Distribution(dist_name=['normal', 'normal'], copula='gumbel')


def pdf(x, params):
    params_marginal_1 = [params[0], params[1]]
    params_marginal_2 = [params[2], params[3]]
    params_copula = params[4]
    return p.pdf(x, params=[params_marginal_1, params_marginal_2], copula_params=params_copula)


def log_pdf(x, params):
    params_marginal_1 = [params[0], params[1]]
    params_marginal_2 = [params[2], params[3]]
    params_copula = params[4]
    return p.log_pdf(x, params=[params_marginal_1, params_marginal_2], copula_params=params_copula)