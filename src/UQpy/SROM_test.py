import scipy.stats as stats
from UQpy import Surrogates
from UQpy import STS

x = STS(dimension=2, pdf_type=['Gamma', 'Gamma'], pdf_params=[[2, 1, 3], [2, 1, 3]], sts_design=[3, 3], pss_=None)


def gamma(z, p):
    return stats.gamma.cdf(z, p[0], loc=p[1], scale=p[2])


# # Creating Correlated Samples
# from scipy import linalg
# self.correlation = np.array(self.correlation)
# l = linalg.cholesky(self.correlation, lower=True)
# self.samples = np.transpose(np.dot(l, np.transpose(self.samples)))

y = Surrogates.SROM(samples=x.samples, pdf_type=[gamma], moments=[[6., 6.], [54., 54.]],
                    properties=[True, True, True, True], pdf_params=[[2, 1, 3]],
                    correlation=[[1, 0], [0, 1]])
