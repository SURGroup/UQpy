from typing import Union
import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionND


class Multinomial(DistributionND):

    @beartype
    def __init__(self, n: Union[None, int], p: Union[list[float], np.ndarray]):
        """

        :param n: number of trials
        :param p: probability of a trial falling into each category; should sum to 1
        """
        super().__init__(n=n, p=p)

    def pmf(self, x):
        pdf_val = stats.multinomial.pmf(x=x, **self.parameters)
        return np.atleast_1d(pdf_val)

    def log_pmf(self, x):
        logpdf_val = stats.multinomial.logpmf(x=x, **self.parameters)
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples=1, random_state=None):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError("Input nsamples must be an integer > 0.")
        return stats.multinomial.rvs(
            size=nsamples, random_state=random_state, **self.parameters
        ).reshape((nsamples, -1))

    def moments(self, moments2return="mv"):
        n = self.get_parameters()["n"]
        p = np.array(self.get_parameters()["p"])
        d = len(p)
        if moments2return == "m":
            mean = n * np.array(p)
            return mean
        elif moments2return == "v":
            cov = (
                -n
                * np.tile(p[np.newaxis, :], [d, 1])
                * np.tile(p[:, np.newaxis], [1, d])
            )
            np.fill_diagonal(cov, n * p * (1.0 - p))
            return cov
        elif moments2return == "mv":
            cov = (
                -n
                * np.tile(p[np.newaxis, :], [d, 1])
                * np.tile(p[:, np.newaxis], [1, d])
            )
            np.fill_diagonal(cov, n * p * (1.0 - p))
            mean = n * np.array(p)
            return mean, cov
        else:
            raise ValueError('UQpy: moments2return must be "m", "v" or "mv".')
