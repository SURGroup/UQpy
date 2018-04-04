from UQpyLibraries import Reliability
from UQpyLibraries import SampleMethods


# x_mcs = SampleMethods.MCS(pdf_type=['Uniform', 'Normal'], pdf_params=[[1, 4], [0, 1]])
#
#
# x_lhs = SampleMethods.LHS(pdf_type=['Uniform', 'Normal'], pdf_params=[[0, 1], [0, 1]], lhs_criterion='correlate')

x_sts = SampleMethods.STS(pdf_type=['Gamma', 'Gamma', 'Gamma'], pdf_params=[[2, 0, 3], [2, 0, 3], [2, 0, 3]], sts_design=[3, 3, 3])


# x_pss = SampleMethods.PSS(pdf_type=['Uniform', 'Normal'], pdf_params=[[0, 1], [0, 1]], pss_design=[1, 1],
#                           pss_strata=[5, 5])

x = SampleMethods.SROM(samples=x_sts.samples, pdf_type=['Gamma', 'Gamma', 'Gamma'], pdf_params=[[2, 0, 3], [2, 0, 3], [2, 0, 3]],
                       moments=[[6, 6, 6], [54, 54, 54]], weights_errors=[1, 0.2, 0.1], properties=[1, 1, 1, 1],
                       correlation=[[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]])

x_mcmc = SampleMethods.MCMC(dimension=2, pdf_proposal_type='Uniform', pdf_target_type='marginal_pdf', skip=10,
                            nsamples=100)

