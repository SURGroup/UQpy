from UQpyLibraries.UQpyModules import *
from UQpyLibraries import SampleMethods
import matplotlib.pyplot as plt

x_mcs = SampleMethods.MCS(pdf_type=['Uniform', 'Normal'], pdf_params=[[1, 4], [0, 1]])


x_lhs = SampleMethods.LHS(pdf_type=['Uniform', 'Normal'], pdf_params=[[0, 1], [0, 1]], lhs_criterion='correlate')

x_sts = SampleMethods.STS(pdf_type=['Uniform', 'Normal'], pdf_params=[[0, 1], [0, 1]], sts_design=[10, 10])


x_pss = SampleMethods.PSS(pdf_type=['Uniform', 'Normal'], pdf_params=[[0, 1], [0, 1]], pss_design=[1, 1],
                          pss_strata=[5, 5])

x_mcmc = SampleMethods.MCMC(dimension=2, pdf_proposal_type='Uniform', pdf_target_type='marginal_pdf', skip=10,
                            nsamples=100)

