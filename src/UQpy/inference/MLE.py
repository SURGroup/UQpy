import logging
from typing import Union

import numpy as np
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer
from beartype import beartype

from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.utilities.Utilities import process_random_state
from UQpy.utilities.ValidationTypes import NumpyFloatArray, RandomStateType


class MLE:
    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    @beartype
    def __init__(
            self,
            inference_model: InferenceModel,
            data: Union[list, np.ndarray],
            n_optimizations: Union[None, int] = 1,
            initial_parameters: Union[list, np.ndarray, None] = None,
            optimizer=MinimizeOptimizer(),
            random_state: RandomStateType = None,
    ):
        """
        Estimate the maximum likelihood parameters of a model given some data.

        :param inference_model: The inference model that defines the likelihood function.
        :param data: Available data, :class:`numpy.ndarray` of shape consistent with log likelihood function in
         :class:`.InferenceModel`. If this parameter is provided at :class:`.InformationModelSelection` object
         initialization, the maximum likelihood estimation algorithm will be automatically performed. Alternatively,
         the user must execute the :meth:`.run` method.
        :param n_optimizations: Number of iterations that the optimization is run, starting at random initial
         guesses. It is only used if `initial_parameters` is not provided. Default is :math:`1`.
         The random initial guesses are sampled uniformly between :math:`0` and :math:`1`, or uniformly between
         user-defined bounds if an input bounds is provided as a keyword argument to the `optimizer` input parameter.
        :param initial_parameters: Initial guess(es) for optimization, :class:`numpy.ndarray` of shape
         :code:`(nstarts, n_parameters)` or :code:`(n_parameters, )`, where :code:`nstarts` is the number of times the
         optimizer will be called. Alternatively, the user can provide input `n_optimizations` to randomly sample
         initial guess(es). The identified MLE is the one that yields the maximum log likelihood over all calls of the
         optimizer.
        :param optimizer: This parameter takes as input an object that implements the :class:`Optimizer` class.
         Default is the :class:`.Minimize` which utilizes the :class:`scipy.optimize.minimize` method.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
        """
        # Initialize variables
        self.inference_model = inference_model
        self.data = data
        self.random_state = process_random_state(random_state)
        self.logger = logging.getLogger(__name__)
        self.optimizer = optimizer
        self.mle: NumpyFloatArray = None
        """Value of parameter vector that maximizes the likelihood function."""
        self.max_log_like: NumpyFloatArray = None
        """Value of the likelihood function at the MLE."""
        self.logger.info("UQpy: Initialization of MLEstimation object completed.")
        self.n_optimizations = n_optimizations
        self.initial_parameters = initial_parameters

        # Run the optimization procedure
        if (n_optimizations is not None) or (initial_parameters is not None):
            self.run(n_optimizations=n_optimizations, initial_parameters=initial_parameters)

    @beartype
    def run(self, n_optimizations: Union[None, int] = 1, initial_parameters: Union[list, np.ndarray, None] = None):
        """
        Run the maximum likelihood estimation procedure.

        This function runs the optimization and updates the :py:attr:`.mle` and :py:attr:`.max_log_like` attributes of
        the class. When learning the parameters of a distribution, if `distributions` possesses an :meth:`mle` method
        this method is used. If `data` are given when creating the :class:`.MLE` object, this method is called
        automatically when the object is created.

        :param n_optimizations: Number of iterations that the optimization is run, starting at random initial
         guesses. It is only used if `initial_parameters` is not provided. Default is :math:`1`.
         The random initial guesses are sampled uniformly between :math:`0` and :math:`1`, or uniformly between
         user-defined bounds if an input bounds is provided as a keyword argument to the `optimizer` input parameter.
        :param initial_parameters: Initial guess(es) for optimization, :class:`numpy.ndarray` of shape
         :code:`(nstarts, n_parameters)` or :code:`(n_parameters, )`, where :code:`nstarts` is the number of times the
         optimizer will be called. Alternatively, the user can provide input `n_optimizations` to randomly sample
         initial guess(es). The identified MLE is the one that yields the maximum log likelihood over all calls of the
         optimizer.
        """
        # Run optimization (use x0 if provided, otherwise sample starting point from [0, 1] or bounds)
        self.logger.info(
            "UQpy: Evaluating maximum likelihood estimate for inference model "
            + self.inference_model.name)

        self.n_optimizations = n_optimizations
        self.initial_parameters = initial_parameters

        use_distribution_fit = (
                hasattr(self.inference_model, "distributions")
                and self.inference_model.distributions is not None
                and hasattr(self.inference_model.distributions, "fit"))

        if use_distribution_fit:
            self._run_distribution_fit(self.n_optimizations)
        else:
            self._run_optimization(self.initial_parameters, self.n_optimizations)

    def _run_distribution_fit(self, n_optimizations):
        for _ in range(n_optimizations):
            self.inference_model.distributions.update_parameters(
                **{key: None for key in self.inference_model.list_params})
            mle_dict = self.inference_model.distributions.fit(data=self.data)
            mle_tmp = np.array([mle_dict[key] for key in self.inference_model.list_params])
            max_log_like_tmp = self.inference_model.evaluate_log_likelihood(
                parameters=mle_tmp[np.newaxis, :], data=self.data)[0]
            # Save result
            if self.mle is None or max_log_like_tmp > self.max_log_like:
                self.mle = mle_tmp
                self.max_log_like = max_log_like_tmp

    def _run_optimization(self, initial_parameters, n_optimizations):
        if initial_parameters is None:
            from UQpy.distributions import Uniform
            initial_parameters = (
                Uniform()
                    .rvs(nsamples=n_optimizations * self.inference_model.n_parameters, random_state=self.random_state, )
                    .reshape((n_optimizations, self.inference_model.n_parameters)))
            if self.optimizer._bounds is not None:
                bounds = np.array(self.optimizer._bounds)
                initial_parameters = (bounds[:, 0].reshape((1, -1))
                                      + (bounds[:, 1] - bounds[:, 0]).reshape((1, -1)) * initial_parameters)
        else:
            initial_parameters = np.atleast_2d(initial_parameters)
            if initial_parameters.shape[1] != self.inference_model.n_parameters:
                raise ValueError("UQpy: Wrong dimensions in x0")
        for x0_ in initial_parameters:
            res = self.optimizer.optimize(self._evaluate_func_to_minimize, x0_)
            mle_tmp = res.x
            max_log_like_tmp = (-1.0) * res.fun
            # Save result
            if self.mle is None or max_log_like_tmp > self.max_log_like:
                self.mle = mle_tmp
                self.max_log_like = max_log_like_tmp
        self.logger.info("UQpy: ML estimation completed.")

    @beartype
    def _evaluate_func_to_minimize(self, one_param: np.ndarray):
        return (-1 * self.inference_model.evaluate_log_likelihood(
            parameters=one_param.reshape((1, -1)), data=self.data)[0])
