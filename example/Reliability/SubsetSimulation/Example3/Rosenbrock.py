from UQpy.Distributions import DistributionND


class Rosenbrock(DistributionND):
    def __init__(self, p=20.):
        super().__init__(p=p)

    def pdf(self, x):
        return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.params['p'])

    def log_pdf(self, x):
        return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.params['p']


# # These functions define the methods of the Rosenbrock bivariate distribution, here pdf and log_pdf.
# # input x must be of the correct dimension, i.e., (nsamples, 2).
# # params must be a list of parameters
#
# import numpy as np
#
#
# # def pdf(x, params):
# #     return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/params[0])
#
#
# # def log_pdf(x, params):
# #     return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/params[0]
#
# def pdf(x, params):
#     return np.exp(-(params[0]*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/20.)
#
#
# def log_pdf(x, params):
#     return -(params[0]*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/20.