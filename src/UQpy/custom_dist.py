import scipy.stats as stats


def gamma_cdf(x, params):

    return stats.gamma.cdf(x, params[0], params[1], params[2])
