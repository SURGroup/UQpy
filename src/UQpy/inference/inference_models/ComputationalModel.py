from UQpy.inference.inference_models.baseclass.InferenceModel import *
from UQpy.distributions.collection.MultivariateNormal import MultivariateNormal


class ComputationalModel(InferenceModel):

    def __init__(self, parameters_number, runmodel_object, error_covariance=1.0, name='', prior=None,
                 log_likelihood=None):
        self.parameters_number = parameters_number
        self.runmodel_object = runmodel_object
        self.error_covariance = error_covariance
        self.name = name
        self.log_likelihood=log_likelihood
        if not isinstance(self.parameters_number, int) or self.parameters_number <= 0:
            raise TypeError('Input nparams must be an integer > 0.')
        self.name = name
        if not isinstance(self.name, str):
            raise TypeError('Input name must be a string.')

        self.prior = prior
        if self.prior is not None:
            if not isinstance(self.prior, Distribution):
                raise TypeError('UQpy: Input prior should be an object of class Distribution.')
            if not hasattr(self.prior, 'log_pdf'):
                if not hasattr(self.prior, 'pdf'):
                    raise AttributeError('UQpy: Input prior should have a log_pdf or pdf method.')
                self.prior.log_pdf = lambda x: np.log(self.prior.pdf(x))

    def evaluate_log_likelihood(self, params, data):

        self.runmodel_object.run(samples=params, append_samples=False)
        model_outputs = self.runmodel_object.qoi_list

        # Case 1.a: Gaussian error model
        if self.log_likelihood is None:
            if isinstance(self.error_covariance, (float, int)):
                norm = Normal(loc=0., scale=np.sqrt(self.error_covariance))
                log_like_values = np.array(
                    [np.sum([norm.log_pdf(data_i - outpt_i) for data_i, outpt_i in zip(data, output)])
                     for output in model_outputs])
            else:
                multivariate_normal = MultivariateNormal(data, cov=self.error_covariance)
                log_like_values = np.array(
                    [multivariate_normal.log_pdf(x=np.array(output).reshape((-1,))) for output in model_outputs])

        # Case 1.b: likelihood is user-defined
        else:
            log_like_values = self.log_likelihood(
                data=data, model_outputs=model_outputs, params=params, **self.kwargs_likelihood)
            if not isinstance(log_like_values, np.ndarray):
                log_like_values = np.array(log_like_values)
            if log_like_values.shape != (params.shape[0],):
                raise ValueError('UQpy: Likelihood function should output a (nsamples, ) ndarray of likelihood '
                                 'values.')
        return log_like_values
