import numpy as np

from UQpy.Distributions import Distribution, Normal, MVNormal
from UQpy.RunModel import RunModel

########################################################################################################################
########################################################################################################################
#                            Define the model - probability model or python model
########################################################################################################################

class InferenceModel:
    """
    Define a probabilistic model for inference.

    **Input:**

    * **nparams** (`int`):
        Number of parameters to be estimated.

    * **name** (`string`):
        Name of model - optional but useful in a model selection setting.

    * **runmodel_object** (object of class ``RunModel``):
        ``RunModel`` class object that defines the forward model. This input is required for cases 1a and 1b.

    * **log_likelihood** (callable):
        Function that defines the log-likelihood model, possibly in conjunction with the `runmodel_object` (cases 1b
        and 2). Default is None, and a Gaussian-error model is considered (case 1a).

        |  If a `runmodel_object` is also defined (case 1b), this function is called as:
        |  `model_outputs = runmodel_object.run(samples=params).qoi_list`
        |  `log_likelihood(params, model_outputs, data, **kwargs_likelihood)`

        |  If no `runmodel_object` is defined (case 2), this function is called as:
        |  `log_likelihood(params, data, **kwargs_likelihood)`

    * **kwargs_likelihood**:
        Keyword arguments transferred to the log-likelihood function.

    * **dist_object** (object of class ``Distribution``):
        Distribution :math:`\pi` for which to learn parameters from iid data (case 3).

        When creating this ``Distribution`` object, the parameters to be learned should be set to `None`.

    * **error_covariance** (`ndarray` or `float`):
        Covariance for Gaussian error model (case 1a). It can be a scalar (in which case the covariance matrix is the
        identity times that value), a 1d `ndarray` in which case the covariance is assumed to be diagonal, or a full
        covariance matrix (2D `ndarray`). Default value is 1.

    * **prior** (object of class ``Distribution``):
        Prior distribution, must have a `log_pdf` or `pdf` method.

    **Methods:**

    """
    # Last Modified: 05/13/2020 by Audrey Olivier

    def __init__(self, nparams, runmodel_object=None, log_likelihood=None, dist_object=None, name='',
                 error_covariance=1.0, prior=None, verbose=False, **kwargs_likelihood
                 ):

        # Initialize some parameters
        self.nparams = nparams
        if not isinstance(self.nparams, int) or self.nparams <= 0:
            raise TypeError('Input nparams must be an integer > 0.')
        self.name = name
        if not isinstance(self.name, str):
            raise TypeError('Input name must be a string.')
        self.verbose = verbose

        self.runmodel_object = runmodel_object
        self.error_covariance = error_covariance
        self.log_likelihood = log_likelihood
        self.dist_object = dist_object
        self.kwargs_likelihood = kwargs_likelihood
        # Perform checks on inputs runmodel_object, log_likelihood, distribution_object that define the inference model
        if (self.runmodel_object is None) and (self.log_likelihood is None) and (self.dist_object is None):
            raise ValueError('UQpy: One of runmodel_object, log_likelihood or dist_object inputs must be provided.')
        if self.runmodel_object is not None and (not isinstance(self.runmodel_object, RunModel)):
            raise TypeError('UQpy: Input runmodel_object should be an object of class RunModel.')
        if (self.log_likelihood is not None) and (not callable(self.log_likelihood)):
            raise TypeError('UQpy: Input log_likelihood should be a callable.')
        if self.dist_object is not None:
            if (self.runmodel_object is not None) or (self.log_likelihood is not None):
                raise ValueError('UQpy: Input dist_object cannot be provided concurrently with log_likelihood '
                                 'or runmodel_object.')
            if not isinstance(self.dist_object, Distribution):
                raise TypeError('UQpy: Input dist_object should be an object of class Distribution.')
            if not hasattr(self.dist_object, 'log_pdf'):
                if not hasattr(self.dist_object, 'pdf'):
                    raise AttributeError('UQpy: dist_object should have a log_pdf or pdf method.')
                self.dist_object.log_pdf = lambda x: np.log(self.dist_object.pdf(x))
            # Check which parameters need to be updated (i.e., those set as None)
            init_params = self.dist_object.get_params()
            self.list_params = [key for key in self.dist_object.order_params if init_params[key] is None]
            if len(self.list_params) != self.nparams:
                raise TypeError('UQpy: Incorrect dimensions between nparams and number of inputs set to None.')

        # Define prior if it is given
        self.prior = prior
        if self.prior is not None:
            if not isinstance(self.prior, Distribution):
                raise TypeError('UQpy: Input prior should be an object of class Distribution.')
            if not hasattr(self.prior, 'log_pdf'):
                if not hasattr(self.prior, 'pdf'):
                    raise AttributeError('UQpy: Input prior should have a log_pdf or pdf method.')
                self.prior.log_pdf = lambda x: np.log(self.prior.pdf(x))

    def evaluate_log_likelihood(self, params, data):
        """
        Evaluate the log likelihood, `log p(data|params)`.

        This method is the central piece of the ``Inference`` module, it is being called repeatedly by all other
        ``Inference`` classes to evaluate the likelihood of the data. The log-likelihood can be evaluated at several
        parameter vectors at once, i.e., `params` is an `ndarray` of shape (nsamples, nparams). If the
        ``InferenceModel`` is powered by ``RunModel`` the ``RunModel.run`` method is called here, possibly leveraging
        its parallel execution.

        **Inputs:**

        * **params** (`ndarray`):
            Parameter vector(s) at which to evaluate the likelihood function, `ndarray` of shape `(nsamples, nparams)`.

        * **data** (`ndarray`):
            Data from which to learn. For case 1b, this should be an `ndarray` of shape `(ndata, )`. For case 3, it must
            be an `ndarray` of shape `(ndata, dimension)`. For other cases it must be consistent with the definition of
            the ``log_likelihood`` callable input.

        **Output/Returns:**

        * (`ndarray`):
            Log-likelihood evaluated at all `nsamples` parameter vector values, `ndarray` of shape (nsamples, ).

        """

        # Check params
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        if len(params.shape) != 2:
            raise TypeError('UQpy: input params should be a nested list or 2d ndarray of shape (nsamples, dimension).')
        if params.shape[1] != self.nparams:
            raise ValueError('UQpy: Wrong dimensions in params.')

        # Case 1 - Forward model is given by RunModel
        if self.runmodel_object is not None:
            self.runmodel_object.run(samples=params, append_samples=False)
            model_outputs = self.runmodel_object.qoi_list

            # Case 1.a: Gaussian error model
            if self.log_likelihood is None:
                if isinstance(self.error_covariance, (float, int)):
                    norm = Normal(loc=0., scale=np.sqrt(self.error_covariance))
                    log_like_values = np.array(
                        [np.sum([norm.log_pdf(data_i-outpt_i) for data_i, outpt_i in zip(data, outpt)])
                         for outpt in model_outputs]
                    )
                else:
                    mvnorm = MVNormal(data, cov=self.error_covariance)
                    log_like_values = np.array(
                        [mvnorm.log_pdf(x=np.array(outpt).reshape((-1,))) for outpt in model_outputs]
                    )

            # Case 1.b: likelihood is user-defined
            else:
                log_like_values = self.log_likelihood(
                    data=data, model_outputs=model_outputs, params=params, **self.kwargs_likelihood
                )
                if not isinstance(log_like_values, np.ndarray):
                    log_like_values = np.array(log_like_values)
                if log_like_values.shape != (params.shape[0],):
                    raise ValueError('UQpy: Likelihood function should output a (nsamples, ) ndarray of likelihood '
                                     'values.')

        # Case 2 - Log likelihood is user defined
        elif self.log_likelihood is not None:
            log_like_values = self.log_likelihood(data=data, params=params, **self.kwargs_likelihood)
            if not isinstance(log_like_values, np.ndarray):
                log_like_values = np.array(log_like_values)
            if log_like_values.shape != (params.shape[0],):
                raise ValueError('UQpy: Likelihood function should output a (nsamples, ) ndarray of likelihood values.')

        # Case 3 - Learn parameters of a probability distribution pi. Data consists in iid sampled from pi.
        else:
            log_like_values = []
            for params_ in params:
                self.dist_object.update_params(**dict(zip(self.list_params, params_)))
                log_like_values.append(np.sum(self.dist_object.log_pdf(x=data)))
            log_like_values = np.array(log_like_values)

        return log_like_values

    def evaluate_log_posterior(self, params, data):
        """
        Evaluate the scaled log posterior `log(p(data|params)p(params))`.

        This method is called by classes that perform Bayesian inference. If the ``InferenceModel`` object does not
        possess a prior, an uninformative prior `p(params)=1` is assumed. Warning: This is an improper prior.

        **Inputs:**

        * **params** (`ndarray`):
            Parameter vector(s) at which to evaluate the log-posterior, `ndarray` of shape (nsamples, nparams).

        * **data** (`ndarray`):
            Data from which to learn. See `evaluate_log_likelihood` method for details.

        **Output/Returns:**

        * (`ndarray`):
            Log-posterior evaluated at all `nsamples` parameter vector values, `ndarray` of shape (nsamples, ).

        """
        # Compute log likelihood
        log_likelihood_eval = self.evaluate_log_likelihood(params=params, data=data)

        # If the prior is not provided it is set to an non-informative prior p(theta)=1, log_posterior = log_likelihood
        if self.prior is None:
            return log_likelihood_eval

        # Otherwise, use prior provided in the InferenceModel setup
        log_prior_eval = self.prior.log_pdf(x=params)

        return log_likelihood_eval + log_prior_eval

