from UQpyLibraries.UQpyModules import *
from UQpyLibraries import SampleMethods
import matplotlib.pyplot as plt

x = SampleMethods.MCS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], nsamples=10)


x = SampleMethods.LHS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], nsamples=10)


x = SampleMethods.STS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], sts_design=[10, 10])

x = SampleMethods.PSS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], pss_design=[2, 2], pss_strata=[5, 5])

# x = SampleMethods.PSS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], pss_design=[2, 2], pss_strata=[5, 5])
#
# x = SampleMethods.MCMC(pdf_target='mvnpdf', mcmc_algorithm='MH', pdf_proposal='Uniform',
#                        pdf_proposal_width=2, pdf_target_params=[[0, 1], [0, 1]])
x= SampleMethods.SROM(samples=x.samples, nsamples=9, marginal=['Uniform', 'Uniform'], moments=[[0, 1], [0, 1]], weights_errors=[1, 0.2, 0],
                 weights_function=None, properties=[1, 1, 0])
print(x.samples)
print()


plt.figure()
plt.scatter(x.samples[:, 0], x.samples[:, 1], marker='.')
plt.show()