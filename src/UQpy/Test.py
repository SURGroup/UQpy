from UQpy.Surrogates import SROM
from UQpy.SampleMethods import STS
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# x = np.random.rand(16, 2)
x = STS(dimension=2, dist_type=['Gamma', 'Gamma'], dist_params=[[2, 1, 3], [2, 1, 3]],
        sts_design=[4, 4], pss_=None)


def gamma_cf(x, params):

    return stats.gamma.pdf(x, params[0], params[1], params[2])


y = SROM(samples=x.samples, dist_type=['gamma_cdf', 'gamma_cdf'], moments=[[6., 6.], [54., 54.]], properties=[True, True, True, False],
         dist_params=[[2, 1, 3], [2, 1, 3]])

# y = SROM(samples=x.samples, pdf_type=[gamma_cf, 'gamma_cdf'], moments=[[6., 6.], [54., 54.]], properties=[True, True, True, False],
#          pdf_params=[[2, 1, 3],[2, 1, 3]], weights_distribution=[[0.4, 0.5]])

# z = SROM(samples=x.samples, pdf_type=['Gamma','Gamma'], moments=[[6., 6.], [54., 54.]], properties=[True, True, True, False],
#          pdf_params=[[2, 1, 3],[2, 1, 3]], weights_moments=[[0.4, 0.5]])