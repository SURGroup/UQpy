from UQpyLibraries.UQpyModules import *
from UQpyLibraries import SampleMethods


x = SampleMethods.MCS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], nsamples=10)


x = SampleMethods.LHS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], nsamples=10)


x = SampleMethods.STS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], sts_design=[10, 10])

x = SampleMethods.PSS(pdf=['Uniform', 'Uniform'], pdf_params=[[0, 1], [0, 1]], pss_design=[2, 2], pss_strata=[5, 5])

x = SampleMethods.MCMC(pdf_target='mvnpdf', mcmc_algorithm='MH', pdf_proposal='Uniform',
                       pdf_proposal_width=2, pdf_target_params=[[0, 1], [0, 1]])

print(x.samples)
model = run_model('./bash_test.sh', 'examples', 'simUQpyOut')
print()
