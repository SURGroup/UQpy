from UQpy.inference.inference_models.baseclass.InferenceModel import *


class DistributionModel(InferenceModel):

    def __init__(self, distributions, nparams, name='', prior=None):
        self.distributions = distributions
        self.nparams = nparams
        self.name = name

        if not isinstance(self.nparams, int) or self.nparams <= 0:
            raise TypeError('Input nparams must be an integer > 0.')

        if not isinstance(self.name, str):
            raise TypeError('Input name must be a string.')

        if self.distributions is not None:
            if not isinstance(self.distributions, Distribution):
                raise TypeError('UQpy: Input dist_object should be an object of class Distribution.')
            if not hasattr(self.distributions, 'log_pdf'):
                if not hasattr(self.distributions, 'pdf'):
                    raise AttributeError('UQpy: dist_object should have a log_pdf or pdf method.')
                self.distributions.log_pdf = lambda x: np.log(self.distributions.pdf(x))
            init_params = self.distributions.get_parameters()
            self.list_params = [key for key in self.distributions.ordered_parameters if init_params[key] is None]
            if len(self.list_params) != self.nparams:
                raise TypeError('UQpy: Incorrect dimensions between nparams and number of inputs set to None.')

        self.prior = prior
        if self.prior is not None:
            if not isinstance(self.prior, Distribution):
                raise TypeError('UQpy: Input prior should be an object of class Distribution.')
            if not hasattr(self.prior, 'log_pdf'):
                if not hasattr(self.prior, 'pdf'):
                    raise AttributeError('UQpy: Input prior should have a log_pdf or pdf method.')
                self.prior.log_pdf = lambda x: np.log(self.prior.pdf(x))



    def evaluate_log_likelihood(self, params, data):
        log_like_values = []
        for params_ in params:
            self.distributions.update_parameters(**dict(zip(self.list_params, params_)))
            log_like_values.append(np.sum(self.distributions.log_pdf(x=data)))
        log_like_values = np.array(log_like_values)
        return log_like_values
