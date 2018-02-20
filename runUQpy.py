# from UQpyLibraries.UQpyModules import *
from UQpyLibraries import SampleMethods
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

#x = SampleMethods.MCS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], nsamples=150)

# x = SampleMethods.LHS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], nsamples=250, lhs_criterion='centered')

#x = SampleMethods.STS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], sts_design=[10, 10])

#x = SampleMethods.PSS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], pss_design=[2, 2], pss_strata=[5, 5])



######## RUN MCMC #########

def mvnorm(x,params):
    return stats.multivariate_normal.pdf(x, mean=params[0], cov=params[1])

def Rosenbrock(x,params):
    return np.exp(-100*((x[1]-x[0]**2)**2+(1-x[0])**2)/20)

x = SampleMethods.MCMC(dim=2, pdf_target=Rosenbrock, mcmc_algorithm='MH', pdf_proposal='Normal',
                      pdf_proposal_params=[1,1])

# x = SampleMethods.MCMC(dim=2, pdf_target=mvnorm, mcmc_algorithm='MH', pdf_proposal='Normal',
#                       pdf_proposal_params=[1,1], pdf_target_params=[[0, 0], [[1, 0], [0, 1]]])


print(x.samples)
print()

plt.figure()
plt.scatter(x.samples[:, 0], x.samples[:, 1], marker='.')
plt.show()