# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This module contains classes and functions for statistical inference from data.

The module currently contains the
following classes:

* ``InferenceModel``: Define a probabilistic model for Inference.
* ``MLEstimation``: Compute maximum likelihood parameter estimate.
* ``InfoModelSelection``: Perform model selection using information theoretic criteria.
* ``BayesParameterEstimation``: Perform Bayesian parameter estimation (estimate posterior density) via MCMC or IS.
* ``BayesModelSelection``: Estimate model posterior probabilities.
"""

import numpy as np

from UQpy.Distributions import Distribution, Normal, MVNormal
from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCMC, IS


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


########################################################################################################################
########################################################################################################################
#                                  Maximum Likelihood Estimation
########################################################################################################################

class MLEstimation:
    """
    Estimate the maximum likelihood parameters of a model given some data.

    **Inputs:**

    * **inference_model** (object of class ``InferenceModel``):
        The inference model that defines the likelihood function.

    * **data** (`ndarray`):
        Available data, `ndarray` of shape consistent with log likelihood function in ``InferenceModel``

    * **optimizer** (callable):
        Optimization algorithm used to compute the mle.

        | This callable takes in as first input the function to be minimized and as second input an initial guess
          (`ndarray` of shape (n_params, )), along with optional keyword arguments if needed, i.e., it is called within
          the code as:
        | `optimizer(func, x0, **kwargs_optimizer)`

        It must return an object with attributes `x` (minimizer) and `fun` (minimum function value).

        Default is `scipy.optimize.minimize`.

    * **kwargs_optimizer**:
        Keyword arguments that will be transferred to the optimizer.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **x0** (`ndarray`):
        Starting point(s) for optimization, see `run_estimation`. Default is `None`.

    * **nopt** (`int`):
        Number of iterations that the optimization is run, starting at random initial guesses. See `run_estimation`.
        Default is `None`.

    If both `x0` and `nopt` are `None`, the object is created but the optimization procedure is not run, one must
    call the ``run`` method.

    **Attributes:**

    * **mle** (`ndarray`):
        Value of parameter vector that maximizes the likelihood function.

    * **max_log_like** (`float`):
        Value of the likelihood function at the MLE.

    **Methods:**

    """
    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier

    def __init__(self, inference_model, data, verbose=False, nopt=None, x0=None, optimizer=None, random_state=None,
                 **kwargs_optimizer):

        # Initialize variables
        self.inference_model = inference_model
        if not isinstance(inference_model, InferenceModel):
            raise TypeError('UQpy: Input inference_model should be of type InferenceModel')
        self.data = data
        self.kwargs_optimizer = kwargs_optimizer
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose
        if optimizer is None:
            from scipy.optimize import minimize
            self.optimizer = minimize
        elif callable(optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError('UQpy: Input optimizer should be None (set to scipy.optimize.minimize) or a callable.')
        self.mle = None
        self.max_log_like = None
        if self.verbose:
            print('UQpy: Initialization of MLEstimation object completed.')

        # Run the optimization procedure
        if (nopt is not None) or (x0 is not None):
            self.run(nopt=nopt, x0=x0)

    def run(self, nopt=1, x0=None):
        """
        Run the maximum likelihood estimation procedure.

        This function runs the optimization and updates the `mle` and `max_log_like` attributes of the class. When
        learning the parameters of a distribution, if `dist_object` possesses an ``mle`` method this method is used. If
        `x0` or `nopt` are given when creating the ``MLEstimation`` object, this method is called automatically when the
        object is created.

        **Inputs:**

        * **x0** (`ndarray`):
            Initial guess(es) for optimization, `ndarray` of shape `(nstarts, nparams)` or `(nparams, )`, where
            `nstarts` is the number of times the optimizer will be called. Alternatively, the user can provide input
            `nopt` to randomly sample initial guess(es). The identified MLE is the one that yields the maximum log
            likelihood over all calls of the optimizer.

        * **nopt** (`int`):
            Number of iterations that the optimization is run, starting at random initial guesses. It is only used if
            `x0` is not provided. Default is 1.

            The random initial guesses are sampled uniformly between 0 and 1, or uniformly between user-defined bounds
            if an input bounds is provided as a keyword argument to the ``MLEstimation`` object.

        """
        # Run optimization (use x0 if provided, otherwise sample starting point from [0, 1] or bounds)
        if self.verbose:
            print('UQpy: Evaluating maximum likelihood estimate for inference model ' + self.inference_model.name)

        # Case 3: check if the distribution pi has a fit method, can be used for MLE. If not, use optimization below.
        if (self.inference_model.dist_object is not None) and hasattr(self.inference_model.dist_object, 'fit'):
            if not (isinstance(nopt, int) and nopt >= 1):
                raise ValueError('UQpy: nopt should be an integer >= 1.')
            for _ in range(nopt):
                self.inference_model.dist_object.update_params(
                    **{key: None for key in self.inference_model.list_params})
                mle_dict = self.inference_model.dist_object.fit(data=self.data)
                mle_tmp = np.array([mle_dict[key] for key in self.inference_model.list_params])
                max_log_like_tmp = self.inference_model.evaluate_log_likelihood(
                    params=mle_tmp[np.newaxis, :], data=self.data)[0]
                # Save result
                if self.mle is None:
                    self.mle = mle_tmp
                    self.max_log_like = max_log_like_tmp
                elif max_log_like_tmp > self.max_log_like:
                    self.mle = mle_tmp
                    self.max_log_like = max_log_like_tmp

        # Otherwise run optimization
        else:
            if x0 is None:
                if not (isinstance(nopt, int) and nopt >= 1):
                    raise ValueError('UQpy: nopt should be an integer >= 1.')
                from UQpy.Distributions import Uniform
                x0 = Uniform().rvs(
                    nsamples=nopt * self.inference_model.nparams, random_state=self.random_state).reshape(
                    (nopt, self.inference_model.nparams))
                if 'bounds' in self.kwargs_optimizer.keys():
                    bounds = np.array(self.kwargs_optimizer['bounds'])
                    x0 = bounds[:, 0].reshape((1, -1)) + (bounds[:, 1] - bounds[:, 0]).reshape((1, -1)) * x0
            else:
                x0 = np.atleast_2d(x0)
                if x0.shape[1] != self.inference_model.nparams:
                    raise ValueError('UQpy: Wrong dimensions in x0')
            for x0_ in x0:
                res = self.optimizer(self._evaluate_func_to_minimize, x0_, **self.kwargs_optimizer)
                mle_tmp = res.x
                max_log_like_tmp = (-1.) * res.fun
                # Save result
                if self.mle is None:
                    self.mle = mle_tmp
                    self.max_log_like = max_log_like_tmp
                elif max_log_like_tmp > self.max_log_like:
                    self.mle = mle_tmp
                    self.max_log_like = max_log_like_tmp

            if self.verbose:
                print('UQpy: ML estimation completed.')

    def _evaluate_func_to_minimize(self, one_param):
        """
        Compute negative log likelihood for one parameter vector.

        This is the function to be minimized in the optimization procedure. This is a utility function that will not be
        called by the user.

        **Inputs:**

        * **one_param** (`ndarray`):
            A parameter vector, `ndarray` of shape (nparams, ).

        **Output/Returns:**

        * (`float`):
            Value of negative log-likelihood.
        """

        return -1 * self.inference_model.evaluate_log_likelihood(params=one_param.reshape((1, -1)), data=self.data)[0]


########################################################################################################################
########################################################################################################################
#                                  Bayesian Parameter estimation
########################################################################################################################

class BayesParameterEstimation:
    """
    Estimate the parameter posterior density given some data.

    This class generates samples from the parameter posterior distribution using Markov Chain Monte Carlo or Importance
    Sampling. It leverages the ``MCMC`` and ``IS`` classes from the ``SampleMethods`` module.


    **Inputs:**

    * **inference_model** (object of class ``InferenceModel``):
        The inference model that defines the likelihood function.

    * **data** (`ndarray`):
        Available data, `ndarray` of shape consistent with log-likelihood function in ``InferenceModel``

    * **sampling_class** (class instance):
        Class instance, must be a subclass of ``MCMC`` or ``IS``.

    * **kwargs_sampler**:
        Keyword arguments of the sampling class, see ``SampleMethods.MCMC`` or ``SampleMethods.IS``.

        Note on the seed for ``MCMC``: if input `seed` is not provided, a seed (`ndarray` of shape
        `(nchains, dimension)`) is sampled from the prior pdf, which must have an `rvs` method.

        Note on the proposal for ``IS``: if no input `proposal` is provided, the prior is used as proposal.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **nsamples** (`int`):
        Number of samples used in MCMC/IS, see `run` method.

    * **samples_per_chain** (`int`):
        Number of samples per chain used in MCMC, see `run` method.

    If both `nsamples` and `nsamples_per_chain` are `None`, the object is created but the sampling procedure is not run,
    one must call the ``run`` method.

    **Attributes:**

    * **sampler** (object of ``SampleMethods`` class specified by `sampling_class`):
        Sampling method object, contains e.g. the posterior samples.

        This object is created along with the ``BayesParameterEstimation`` object, and its `run` method is called
        whenever the `run` method of the ``BayesParameterEstimation`` is called.

    **Methods:**

    """
    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    def __init__(self, inference_model, data, sampling_class=None, nsamples=None, nsamples_per_chain=None,
                 random_state=None, verbose=False, **kwargs_sampler):

        self.inference_model = inference_model
        if not isinstance(self.inference_model, InferenceModel):
            raise TypeError('UQpy: Input inference_model should be of type InferenceModel')
        self.data = data
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        from UQpy.SampleMethods import MCMC, IS
        # MCMC algorithm
        if issubclass(sampling_class, MCMC):
            # If the seed is not provided, sample one from the prior pdf of the parameters
            if 'seed' not in kwargs_sampler.keys() or kwargs_sampler['seed'] is None:
                if self.inference_model.prior is None or not hasattr(self.inference_model.prior, 'rvs'):
                    raise NotImplementedError('UQpy: A prior with a rvs method or a seed must be provided for MCMC.')
                else:
                    kwargs_sampler['seed'] = self.inference_model.prior.rvs(
                        nsamples=kwargs_sampler['nchains'], random_state=self.random_state)
            self.sampler = sampling_class(
                dimension=self.inference_model.nparams, verbose=self.verbose, random_state=self.random_state,
                log_pdf_target=self.inference_model.evaluate_log_posterior, args_target=(self.data, ),
                nsamples=None, nsamples_per_chain=None, **kwargs_sampler)

        elif issubclass(sampling_class, IS):
            # Importance distribution is either given by the user, or it is set as the prior of the model
            if 'proposal' not in kwargs_sampler or kwargs_sampler['proposal'] is None:
                if self.inference_model.prior is None:
                    raise NotImplementedError('UQpy: A proposal density or a prior must be provided.')
                kwargs_sampler['proposal'] = self.inference_model.prior

            self.sampler = sampling_class(
                log_pdf_target=self.inference_model.evaluate_log_posterior, args_target=(self.data, ),
                random_state=self.random_state, verbose=self.verbose, nsamples=None, **kwargs_sampler)

        else:
            raise ValueError('UQpy: Sampling_class should be either a MCMC algorithm or IS.')

        # Run the analysis if a certain number of samples was provided
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run(self, nsamples=None, nsamples_per_chain=None):
        """
        Run the Bayesian inference procedure, i.e., sample from the parameter posterior distribution.

        This function calls the ``run`` method of the `sampler` attribute to generate samples from the parameter
        posterior distribution.

        **Inputs:**

        * **nsamples** (`int`):
            Number of samples used in ``MCMC``/``IS``

        * **samples_per_chain** (`int`):
            Number of samples per chain used in ``MCMC``

        """

        if isinstance(self.sampler, MCMC):
            self.sampler.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

        elif isinstance(self.sampler, IS):
            if nsamples_per_chain is not None:
                raise ValueError('UQpy: nsamples_per_chain is not an appropriate input for IS.')
            self.sampler.run(nsamples=nsamples)

        else:
            raise ValueError('UQpy: sampling class should be a subclass of MCMC or IS')

        if self.verbose:
            print('UQpy: Parameter estimation with ' + self.sampler.__class__.__name__ + ' completed successfully!')


########################################################################################################################
########################################################################################################################
#                                  Model Selection Using Information Theoretic Criteria
########################################################################################################################

class InfoModelSelection:
    """
    Perform model selection using information theoretic criteria.

    Supported criteria are BIC, AIC (default), AICc. This class leverages the ``MLEstimation`` class for maximum
    likelihood estimation, thus inputs to ``MLEstimation`` can also be provided to ``InfoModelSelection``, as lists of
    length equal to the number of models.


    **Inputs:**

    * **candidate_models** (`list` of ``InferenceModel`` objects):
        Candidate models

    * **data** (`ndarray`):
        Available data

    * **criterion** (`str`):
        Criterion to be used ('AIC', 'BIC', 'AICc'). Default is 'AIC'

    * **kwargs**:
        Additional keyword inputs to the maximum likelihood estimators.

        Keys must refer to input names to the ``MLEstimation`` class, and values must be lists of length `nmodels`,
        ordered in the same way as input `candidate_models`. For example, setting
        `kwargs={`method': [`Nelder-Mead', `Powell']}` means that the Nelder-Mead minimization algorithm will be used
        for ML estimation of the first candidate model, while the Powell method will be used for the second candidate
        model.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **x0** (`list` of `ndarrays`):
        Starting points for optimization - see ``MLEstimation``

    * **nopt** (`list` of `int`):
        Number of iterations for the maximization procedure - see ``MLEstimation``

    If `x0` and `nopt` are both `None`, the object is created but the model selection procedure is not run, one
    must then call the ``run`` method.

    **Attributes:**

    * **ml_estimators** (`list` of `MLEstimation` objects):
        ``MLEstimation`` results for each model (contains e.g. fitted parameters)

    * **criterion_values** (`list` of `floats`):
        Value of the criterion for all models.

    * **penalty_terms** (`list` of `floats`):
        Value of the penalty term for all models. Data fit term is then criterion_value - penalty_term.

    * **probabilities** (`list` of `floats`):
        Value of the model probabilities, computed as

        .. math:: P(M_i|d) = \dfrac{\exp(-\Delta_i/2)}{\sum_i \exp(-\Delta_i/2)}

        where :math:`\Delta_i = criterion_i - min_i(criterion)`

    **Methods:**

    """
    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    def __init__(self, candidate_models, data, criterion='AIC', random_state=None, verbose=False, nopt=None, x0=None,
                 **kwargs):

        # Check inputs
        # candidate_models is a list of InferenceModel objects
        if not isinstance(candidate_models, (list, tuple)) or not all(isinstance(model, InferenceModel)
                                                                      for model in candidate_models):
            raise TypeError('UQpy: Input candidate_models must be a list of InferenceModel objects.')
        self.nmodels = len(candidate_models)
        self.candidate_models = candidate_models
        self.data = data
        if criterion not in ['AIC', 'BIC', 'AICc']:
            raise ValueError('UQpy: Criterion should be AIC (default), BIC or AICc')
        self.criterion = criterion
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        # Instantiate the ML estimators
        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or \
                not all(len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            raise TypeError('UQpy: Extra inputs to model selection must be lists of length len(candidate_models)')
        self.ml_estimators = []
        for i, inference_model in enumerate(self.candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in kwargs.items()])
            ml_estimator = MLEstimation(inference_model=inference_model, data=self.data, verbose=self.verbose,
                                        random_state=self.random_state, x0=None, nopt=None, **kwargs_i)
            self.ml_estimators.append(ml_estimator)

        # Initialize the outputs
        self.criterion_values = [None, ] * self.nmodels
        self.penalty_terms = [None, ] * self.nmodels
        self.probabilities = [None, ] * self.nmodels

        # Run the model selection procedure
        if (nopt is not None) or (x0 is not None):
            self.run(nopt=nopt, x0=x0)

    def run(self, nopt=1, x0=None):
        """
        Run the model selection procedure, i.e. compute criterion value for all models.

        This function calls the ``run`` method of the ``MLEstimation`` object for each model to compute the maximum
        log-likelihood, then computes the criterion value and probability for each model.

        **Inputs:**

        * **x0** (`list` of `ndarrays`):
            Starting point(s) for optimization for all models. Default is `None`. If not provided, see `nopt`. See
            ``MLEstimation`` class.

        * **nopt** (`int` or `list` of `ints`):
            Number of iterations that the optimization is run, starting at random initial guesses. It is only used if
            `x0` is not provided. Default is 1. See ``MLEstimation`` class.

        """
        # Check inputs x0, nopt
        if isinstance(nopt, int) or nopt is None:
            nopt = [nopt] * self.nmodels
        if not (isinstance(nopt, list) and len(nopt) == self.nmodels):
            raise ValueError('UQpy: nopt should be an int or list of length nmodels')
        if x0 is None:
            x0 = [None] * self.nmodels
        if not (isinstance(x0, list) and len(x0) == self.nmodels):
            raise ValueError('UQpy: x0 should be a list of length nmodels (or None).')

        # Loop over all the models
        for i, (inference_model, ml_estimator) in enumerate(zip(self.candidate_models, self.ml_estimators)):
            # First evaluate ML estimate for all models, do several iterations if demanded
            ml_estimator.run(nopt=nopt[i], x0=x0[i])

            # Then minimize the criterion
            self.criterion_values[i], self.penalty_terms[i] = self._compute_info_criterion(
                criterion=self.criterion, data=self.data, inference_model=inference_model,
                max_log_like=ml_estimator.max_log_like, return_penalty=True)

        # Compute probabilities from criterion values
        self.probabilities = self._compute_probabilities(self.criterion_values)

    def sort_models(self):
        """
        Sort models in descending order of model probability (increasing order of `criterion` value).

        This function sorts - in place - the attribute lists `candidate_models, ml_estimators, criterion_values,
        penalty_terms` and `probabilities` so that they are sorted from most probable to least probable model. It is a
        stand-alone function that is provided to help the user to easily visualize which model is the best.

        No inputs/outputs.

        """
        sort_idx = list(np.argsort(np.array(self.criterion_values)))

        self.candidate_models = [self.candidate_models[i] for i in sort_idx]
        self.ml_estimators = [self.ml_estimators[i] for i in sort_idx]
        self.criterion_values = [self.criterion_values[i] for i in sort_idx]
        self.penalty_terms = [self.penalty_terms[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]

    @staticmethod
    def _compute_info_criterion(criterion, data, inference_model, max_log_like, return_penalty=False):
        """
        Compute the criterion value for a given model, given a max_log_likelihood value.

        The criterion value is -2 * max_log_like + penalty, the penalty depends on the chosen criterion. This function
        is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param criterion: Chosen criterion.
        :type criterion: str

        :param data: Available data.
        :type data: ndarray

        :param inference_model: Inference model.
        :type inference_model: object of class InferenceModel

        :param max_log_like: Value of likelihood function at MLE.
        :type max_log_like: float

        :param return_penalty: Boolean that sets whether to return the penalty term as additional output.

                               Default is False
        :type return_penalty: bool

        **Output/Returns:**

        :return criterion_value: Value of criterion.
        :rtype criterion_value: float

        :return penalty_term: Value of penalty term.
        :rtype penalty_term: float

        """

        n_params = inference_model.nparams
        ndata = len(data)
        if criterion == 'BIC':
            penalty_term = np.log(ndata) * n_params
        elif criterion == 'AICc':
            penalty_term = 2 * n_params + (2 * n_params ** 2 + 2 * n_params) / (ndata - n_params - 1)
        elif criterion == 'AIC':  # default
            penalty_term = 2 * n_params
        else:
            raise ValueError('UQpy: Criterion should be AIC (default), BIC or AICc')
        if return_penalty:
            return -2 * max_log_like + penalty_term, penalty_term
        return -2 * max_log_like + penalty_term

    @staticmethod
    def _compute_probabilities(criterion_values):
        """
        Compute the model probability given criterion values for all models.

        Model probability is proportional to exp(-criterion/2), model probabilities over all models sum up to 1. This
        function is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param criterion_values: Values of criterion for all models.
        :type criterion_values: list (length nmodels) of floats

        **Output/Returns:**

        :return probabilities: Values of model probabilities
        :rtype probabilities: list (length nmodels) of floats

        """

        delta = np.array(criterion_values) - min(criterion_values)
        prob = np.exp(-delta / 2)
        return prob / np.sum(prob)


########################################################################################################################
########################################################################################################################
#                                  Bayesian Model Selection
########################################################################################################################


class BayesModelSelection:

    """
    Perform model selection via Bayesian inference, i.e., compute model posterior probabilities given data.

    This class leverages the ``BayesParameterEstimation`` class to get samples from the parameter posterior densities.
    These samples are then used to compute the model evidence `p(data|model)` for all models and the model posterior
    probabilities.

    **References:**

    1. A.E. Raftery, M.A. Newton, J.M. Satagopan, and P.N. Krivitsky. "Estimating the integrated likelihood via
       posterior simulation using the harmonic mean identity". In Bayesian Statistics 8, pages 1–45, 2007.

    **Inputs:**

    * **candidate_models** (`list` of ``InferenceModel`` objects):
        Candidate models

    * **data** (`ndarray`):
        Available data

    * **prior_probabilities** (`list` of `floats`):
        Prior probabilities of each model, default is [1/nmodels, ] * nmodels

    * **method_evidence_computation** (`str`):
        as of v3, only the harmonic mean method is supported

    * **kwargs**:
        Keyword arguments to the ``BayesParameterEstimation`` class, for each model.

        Keys must refer to names of inputs to the ``MLEstimation`` class, and values should be lists of length
        `nmodels`, ordered in the same way as input candidate_models. For example, setting
        `kwargs={`sampling_class': [MH, Stretch]}` means that the MH algorithm will be used for sampling from the
        parameter posterior pdf of the 1st candidate model, while the Stretch algorithm will be used for the 2nd model.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **nsamples** (`list` of `int`):
        Number of samples used in ``MCMC``/``IS``, for each model

    * **samples_per_chain** (`list` of `int`):
        Number of samples per chain used in ``MCMC``, for each model

    If `nsamples` and `nsamples_per_chain` are both `None`, the object is created but the model selection procedure is
    not run, one must then call the ``run`` method.

    **Attributes:**

    * **bayes_estimators** (`list` of ``BayesParameterEstimation`` objects):
        Results of the Bayesian parameter estimation

    * **self.evidences** (`list` of `floats`):
        Value of the evidence for all models

    * **probabilities** (`list` of `floats`):
        Posterior probability for all models

    **Methods:**

    """
    # Authors: Audrey Olivier, Yuchen Zhou
    # Last modified: 01/24/2020 by Audrey Olivier
    def __init__(self, candidate_models, data, prior_probabilities=None, method_evidence_computation='harmonic_mean',
                 random_state=None, verbose=False, nsamples=None, nsamples_per_chain=None, **kwargs):

        # Check inputs: candidate_models is a list of instances of Model, data must be provided, and input arguments
        # for MCMC must be provided as a list of length len(candidate_models)
        if (not isinstance(candidate_models, list)) or (not all(isinstance(model, InferenceModel)
                                                                for model in candidate_models)):
            raise TypeError('UQpy: A list InferenceModel objects must be provided.')
        self.candidate_models = candidate_models
        self.nmodels = len(candidate_models)
        self.data = data
        self.method_evidence_computation = method_evidence_computation
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        if prior_probabilities is None:
            self.prior_probabilities = [1. / len(candidate_models) for _ in candidate_models]
        else:
            self.prior_probabilities = prior_probabilities

        # Instantiate the Bayesian parameter estimators (without running them)
        self.bayes_estimators = []
        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or not all(
                len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            raise TypeError('UQpy: Extra inputs to model selection must be lists of length len(candidate_models)')
        for i, inference_model in enumerate(self.candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in kwargs.items()])
            kwargs_i.update({'concat_chains': True, 'save_log_pdf': True})
            bayes_estimator = BayesParameterEstimation(
                inference_model=inference_model, data=self.data, verbose=self.verbose,
                random_state=self.random_state, nsamples=None, nsamples_per_chain=None, **kwargs_i)
            self.bayes_estimators.append(bayes_estimator)

        # Initialize the outputs
        self.evidences = [0.] * self.nmodels
        self.probabilities = [0.] * self.nmodels

        # Run the model selection procedure
        if nsamples is not None or nsamples_per_chain is not None:
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run(self, nsamples=None, nsamples_per_chain=None):
        """
        Run the Bayesian model selection procedure, i.e., compute model posterior probabilities.

        This function calls the ``run_estimation`` method of the ``BayesParameterEstimation`` object for each model to
        sample from the parameter posterior probability, then computes the model evidence and model posterior
        probability. This function updates attributes `bayes_estimators`, `evidences` and `probabilities`. If `nsamples`
        or `nsamples_per_chain` are given when creating the object, this method is called directly when the object is
        created. It can also be called separately.

        **Inputs:**

        * **nsamples** (`list` of `int`):
            Number of samples used in ``MCMC``/``IS``, for each model

        * **samples_per_chain** (`list` of `int`):
            Number of samples per chain used in ``MCMC``, for each model

        """

        if nsamples is not None and not (isinstance(nsamples, list) and len(nsamples) == self.nmodels
                                         and all(isinstance(n, int) for n in nsamples)):
            raise ValueError('UQpy: nsamples should be a list of integers')
        if nsamples_per_chain is not None and not (isinstance(nsamples_per_chain, list)
                                                   and len(nsamples_per_chain) == self.nmodels
                                                   and all(isinstance(n, int) for n in nsamples_per_chain)):
            raise ValueError('UQpy: nsamples_per_chain should be a list of integers')
        if self.verbose:
            print('UQpy: Running Bayesian Model Selection.')
        # Perform MCMC for all candidate models
        for i, (inference_model, bayes_estimator) in enumerate(zip(self.candidate_models, self.bayes_estimators)):
            if self.verbose:
                print('UQpy: Running MCMC for model '+inference_model.name)
            if nsamples is not None:
                bayes_estimator.run(nsamples=nsamples[i])
            elif nsamples_per_chain is not None:
                bayes_estimator.run(nsamples_per_chain=nsamples_per_chain[i])
            else:
                raise ValueError('UQpy: ither nsamples or nsamples_per_chain should be non None')
            self.evidences[i] = self._estimate_evidence(
                method_evidence_computation=self.method_evidence_computation,
                inference_model=inference_model, posterior_samples=bayes_estimator.sampler.samples,
                log_posterior_values=bayes_estimator.sampler.log_pdf_values)

        # Compute posterior probabilities
        self.probabilities = self._compute_posterior_probabilities(
            prior_probabilities=self.prior_probabilities, evidence_values=self.evidences)

        if self.verbose:
            print('UQpy: Bayesian Model Selection analysis completed!')

    def sort_models(self):
        """
        Sort models in descending order of model probability (increasing order of criterion value).

        This function sorts - in place - the attribute lists `candidate_models`, `prior_probabilities`, `probabilities`
        and `evidences` so that they are sorted from most probable to least probable model. It is a stand-alone function
        that is provided to help the user to easily visualize which model is the best.

        No inputs/outputs.

        """
        sort_idx = list(np.argsort(np.array(self.probabilities)))[::-1]

        self.candidate_models = [self.candidate_models[i] for i in sort_idx]
        self.prior_probabilities = [self.prior_probabilities[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]
        self.evidences = [self.evidences[i] for i in sort_idx]

    @staticmethod
    def _estimate_evidence(method_evidence_computation, inference_model, posterior_samples, log_posterior_values):
        """
        Compute the model evidence, given samples from the parameter posterior pdf.

        As of V3, only the harmonic mean method is supported for evidence computation. This function
        is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param method_evidence_computation: Method for evidence computation. As of v3, only the harmonic mean is
                                            supported.
        :type method_evidence_computation: str

        :param inference_model: Inference model.
        :type inference_model: object of class InferenceModel

        :param posterior_samples: Samples from parameter posterior density.
        :type posterior_samples: ndarray of shape (nsamples, nparams)

        :param log_posterior_values: Log-posterior values of the posterior samples.
        :type log_posterior_values: ndarray of shape (nsamples, )

        **Output/Returns:**

        :return evidence: Value of evidence p(data|M).
        :rtype evidence: float

        """
        if method_evidence_computation.lower() == 'harmonic_mean':
            # samples[int(0.5 * len(samples)):]
            log_likelihood_values = log_posterior_values - inference_model.prior.log_pdf(x=posterior_samples)
            temp = np.mean(1./np.exp(log_likelihood_values))
        else:
            raise ValueError('UQpy: Only the harmonic mean method is currently supported')
        return 1./temp

    @staticmethod
    def _compute_posterior_probabilities(prior_probabilities, evidence_values):
        """
        Compute the model probability given prior probabilities P(M) and evidence values p(data|M).

        Model posterior probability P(M|data) is proportional to p(data|M)P(M). Posterior probabilities sum up to 1 over
        all models. This function is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param prior_probabilities: Values of prior probabilities for all models.
        :type prior_probabilities: list (length nmodels) of floats

        :param prior_probabilities: Values of evidence for all models.
        :type prior_probabilities: list (length nmodels) of floats

        **Output/Returns:**

        :return probabilities: Values of model posterior probabilities
        :rtype probabilities: list (length nmodels) of floats

        """
        scaled_evidences = [evi * prior_prob for (evi, prior_prob) in zip(evidence_values, prior_probabilities)]
        return scaled_evidences / np.sum(scaled_evidences)
