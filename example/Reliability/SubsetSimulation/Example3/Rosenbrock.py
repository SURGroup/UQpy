from UQpy.distributions import DistributionND
import numpy as np

class Rosenbrock(DistributionND):
    def __init__(self, p=20.):
        super().__init__(p=p)

    def pdf(self, x):
        return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.params['trial_probability'])

    def log_pdf(self, x):
        return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.params['trial_probability']


# # These functions define the methods of the Rosenbrock bivariate distribution, here pdf and log_pdf.
# # input x must be of the correct dimension, i.e., (nsamples, 2).
# # parameter_vector must be a list of parameters
#
# import numpy as np
#
#
# # def pdf(x, parameter_vector):
# #     return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/parameter_vector[0])
#
#
# # def log_pdf(x, parameter_vector):
# #     return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/parameter_vector[0]
#
# def pdf(x, parameter_vector):
#     return np.exp(-(parameter_vector[0]*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/20.)
#
#
# def log_pdf(x, parameter_vector):
#     return -(parameter_vector[0]*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/20.