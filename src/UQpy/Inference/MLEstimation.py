import numpy as np

from UQpy.Inference.InferenceModel import InferenceModel


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


