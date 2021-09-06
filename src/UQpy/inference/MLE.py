import logging
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.inference.inference_models.optimization.Optimizer import Optimizer
from UQpy.inference.inference_models.optimization.MinizeOptimizer import MinimizeOptimizer
from UQpy.utilities.Utilities import process_random_state
from UQpy.utilities.ValidationTypes import PositiveInteger


class MLE:
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
    @beartype
    def __init__(self,
                 inference_model: InferenceModel,
                 data: Union[list, np.ndarray],
                 optimizations_number: Union[None, int] = None,
                 initial_guess=None,
                 optimizer: Optimizer = MinimizeOptimizer(),
                 random_state=None):

        # Initialize variables
        self.inference_model = inference_model
        self.data = data
        self.random_state = process_random_state(random_state)
        self.logger = logging.getLogger(__name__)
        self.optimizer = optimizer
        self.mle = None
        self.max_log_like = None
        self.logger.info('UQpy: Initialization of MLEstimation object completed.')

        # Run the optimization procedure
        if (optimizations_number is not None) or (initial_guess is not None):
            self.run(optimizations_number=optimizations_number, initial_guess=initial_guess)

    @beartype
    def run(self, optimizations_number: Union[None, PositiveInteger] = 1, initial_guess=None):
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
        self.logger.info('UQpy: Evaluating maximum likelihood estimate for inference model '
                         + self.inference_model.name)

        use_distribution_fit = hasattr(self.inference_model, 'distributions') and \
                               self.inference_model.distributions is not None and \
                               hasattr(self.inference_model.distributions, 'fit')

        if use_distribution_fit:
            self._run_distribution_fit(optimizations_number)
        else:
            self._run_optimization(initial_guess, optimizations_number)

    def _run_distribution_fit(self, optimizations_number):
        for _ in range(optimizations_number):
            self.inference_model.distributions.update_parameters(
                **{key: None for key in self.inference_model.list_params})
            mle_dict = self.inference_model.distributions.fit(data=self.data)
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

    def _run_optimization(self, initial_guess, optimizations_number):
        if initial_guess is None:
            from UQpy.distributions import Uniform
            initial_guess = Uniform().rvs(
                nsamples=optimizations_number * self.inference_model.parameters_number,
                random_state=self.random_state) \
                .reshape((optimizations_number, self.inference_model.parameters_number))
            if self.optimizer.bounds is not None:
                bounds = np.array(self.optimizer.bounds)
                initial_guess = bounds[:, 0].reshape((1, -1)) + (bounds[:, 1] - bounds[:, 0]).reshape((1, -1)) \
                                * initial_guess
        else:
            initial_guess = np.atleast_2d(initial_guess)
            if initial_guess.shape[1] != self.inference_model.parameters_number:
                raise ValueError('UQpy: Wrong dimensions in x0')
        for x0_ in initial_guess:
            res = self.optimizer.optimize(self._evaluate_func_to_minimize, x0_)
            mle_tmp = res.x
            max_log_like_tmp = (-1.) * res.fun
            # Save result
            if self.mle is None:
                self.mle = mle_tmp
                self.max_log_like = max_log_like_tmp
            elif max_log_like_tmp > self.max_log_like:
                self.mle = mle_tmp
                self.max_log_like = max_log_like_tmp
        self.logger.info('UQpy: ML estimation completed.')

    @beartype
    def _evaluate_func_to_minimize(self, one_param: np.ndarray):
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

        a= -1 * self.inference_model.evaluate_log_likelihood(params=one_param.reshape((1, -1)), data=self.data)[0]
        return a
