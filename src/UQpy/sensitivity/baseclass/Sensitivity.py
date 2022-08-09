"""

This module contains the abstract Sensitivity class used by other 
sensitivity classes: 
1. Chatterjee indices
2. Cramer-von Mises indices
3. Generalised Sobol indices
4. Sobol indices

"""

import copy
import numpy as np
import scipy.stats

from typing import Union
from beartype import beartype

from UQpy.distributions import *
from UQpy.utilities.ValidationTypes import (
    PositiveFloat,
    RandomStateType,
    PositiveInteger,
    NumpyFloatArray,
    NumpyIntArray,
)
from UQpy.run_model import RunModel
from UQpy.distributions.collection import JointIndependent


class Sensitivity:
    @beartype
    def __init__(
        self,
        runmodel_object: RunModel,
        dist_object: Union[JointIndependent, Union[list, tuple]],
        random_state: RandomStateType = None
    ) -> None:

        self.runmodel_object = runmodel_object
        self.dist_object = dist_object
        self.random_state = random_state

    # wrapper created for convenience to generate model evaluations
    @beartype
    def _run_model(self, samples: Union[NumpyFloatArray, NumpyIntArray]):
        """Generate model evaluations for a set of samples.

        **Inputs**:

        * **samples** (`numpy.ndarray`):
            A set of samples.
            Shape: `(n_samples, num_vars)`

        **Outputs**:

        * **model_evaluations** (`numpy.ndarray`):
            A set of model evaluations.
            Shape: `(n_samples,)`

            if multioutput: `(n_samples, n_outputs)`

        """

        self.runmodel_object.run(samples=samples, append_samples=False)
        model_evals = copy.deepcopy(np.array(self.runmodel_object.qoi_list))

        return model_evals

    @staticmethod
    @beartype
    def bootstrap_sample_generator_1D(samples: Union[NumpyFloatArray, NumpyIntArray]):
        """Generate bootstrap samples.

        Generators are used to avoid copying the entire array.

        It will simply pick `N` random rows from the array.

        For example:
        Model evaluations for the samples in A in the pick and freeze estimator.

        **Inputs:**

        * **samples** (`ndarray`):
            Model evaluations for the samples.
            Shape: `(n_samples, 1)`.

        **Outputs:**

        * `generator`:
            Generator for the bootstrap samples.

        """
        n_samples = samples.shape[0]

        while True:
            _indices = np.random.randint(0, high=n_samples, size=n_samples)

            yield samples[_indices]

    @staticmethod
    @beartype
    def bootstrap_sample_generator_2D(samples: Union[NumpyFloatArray, NumpyIntArray]):
        """Generate bootstrap samples.

        Generators are used to avoid copying the entire array.

        For example:
        Let's say we have '3' random variables
        To pick bootstrap samples from f_C_i, we first
        generate indices to pick values from each column
        num_cols = 3
        cols = [0, 1, 2]
        _indices = [[3, 4, 8]
                    [6, 1, 2]
                    [0, 5, 7]
                    [4, 1, 0]] (4x3)
        elements from f_C_i will be picked column-wise:
        f_C_i[_indices[:, 0], 0]
        f_C_i[_indices[:, 1], 1] etc.

        **Inputs:**

        * **samples** (`ndarray`):
            Model evaluations for the samples.
            Shape: `(n_samples, 1)`.

        **Outputs:**

        * `generator`:
            Generator for the bootstrap samples.

        """
        n_samples = samples.shape[0]

        num_cols = samples.shape[1]
        cols = np.arange(num_cols)

        while True:
            # generate indices to pick N values from f_A, f_B and f_C_i
            _indices = np.random.randint(0, high=n_samples, size=samples.shape)

            yield samples[_indices, cols]

    @staticmethod
    @beartype
    def bootstrap_sample_generator_3D(samples: Union[NumpyFloatArray, NumpyIntArray]):
        """Generate bootstrap samples.

        Generators are used to avoid copying the entire array.

        For example:
        Let's say we a model with multiple outputs.
        We use the same approach as in the 2D
        case for each slice the 3D array.
        Here, slices refer to the 'depth' of the array,
        given by array.shape[0].

        **Inputs:**

        * **samples** (`ndarray`):
            Model evaluations for the samples.
            Shape: `(n_outputs, n_samples, num_vars)`.

        **Outputs:**

        * `generator`:
            Generator for the bootstrap samples.

        """
        n_samples = samples.shape[1]
        array_shape = samples.shape[1:]
        num_cols = samples.shape[2]
        cols = np.arange(num_cols)

        while True:
            _indices = np.random.randint(0, high=n_samples, size=array_shape)

            yield samples[:, _indices, cols]

    @beartype
    def bootstrapping(
        self,
        estimator,
        estimator_inputs,
        qoi_mean: Union[NumpyFloatArray, NumpyIntArray],
        num_bootstrap_samples: PositiveInteger = None,
        confidence_level: PositiveFloat = 0.95,
        **kwargs,
    ):

        """An abstract method to implement bootstrapping.

        **Inputs:**

        * **estimator** (`function`):
            A method/func which computes the statistical
            quantities of interest (QoI).
            Example: `compute_first_order_Sobol`
            It must be a method/function that takes several `ndarray`s
            of samples as input and returns a single `ndarray` of estimated value.

        * **estimator_inputs** (`list`):
            Inputs to the estimator concantenated in a list.

        * **qoi_mean** (`ndarray`):
            Mean of the QoI.
            This is the value around which we
            will compute the confidence interval.
            Shape: `(n_qois, n_outputs)`.

        * **num_bootstrap_samples** (`int`):
            Number of bootstrap samples to generate.

        * **confidence_level** (`float`):
            Confidence level for the confidence interval.
            Default: 0.95

        **Outputs:**

        * **confidence_interval_qoi** (`ndarray`):
            Confidence interval for the quantity of interest (QoI).

        """

        n_qois = qoi_mean.shape[0]
        n_outputs = qoi_mean.shape[1]

        ##################### STORAGE #####################

        # store generators of the inputs for bootstrap sampling
        input_generators = []

        # store the qoi computed using bootstrap samples
        bootstrapped_qoi = np.zeros((n_outputs, n_qois, num_bootstrap_samples))

        # store the confidence interval for each qoi
        confidence_interval_qoi = np.zeros((n_outputs, n_qois, 2))

        self._create_generators(estimator_inputs, input_generators)

        self._evaluate_boostrap_sample_qoi(bootstrapped_qoi, estimator, input_generators, kwargs, num_bootstrap_samples)

        confidence_interval_qoi = self._calculate_confidence_intervals(bootstrapped_qoi, confidence_interval_qoi,
                                                                       confidence_level, n_outputs, qoi_mean)

        return confidence_interval_qoi

    def _evaluate_boostrap_sample_qoi(self, bootstrapped_qoi, estimator, input_generators, kwargs,
                                      num_bootstrap_samples):
        # Compute the qoi for each bootstrap sample
        for j in range(num_bootstrap_samples):

            # inputs to the estimator
            args = []

            # generate samples
            for gen_input in input_generators:
                if gen_input is None:
                    args.append(gen_input)
                else:
                    args.append(gen_input.__next__())

            bootstrapped_qoi[:, :, j] = estimator(*args, **kwargs).T

    def _calculate_confidence_intervals(self, bootstrapped_qoi, confidence_interval_qoi, confidence_level, n_outputs,
                                        qoi_mean):
        # Calculate confidence intervals
        delta = -scipy.stats.norm.ppf((1 - confidence_level) / 2)
        for output_j in range(n_outputs):
            # estimate the standard deviation using the bootstrap indices
            std_qoi = np.std(bootstrapped_qoi[output_j, :, :], axis=1, ddof=1)

            lower_bound = qoi_mean[:, output_j] - delta * std_qoi
            upper_bound = qoi_mean[:, output_j] + delta * std_qoi

            confidence_interval_qoi[output_j, :, 0] = lower_bound
            confidence_interval_qoi[output_j, :, 1] = upper_bound
        # For models with single output, return 2D array.
        if n_outputs == 1:
            confidence_interval_qoi = confidence_interval_qoi[0, :, :]
        return confidence_interval_qoi

    def _create_generators(self, estimator_inputs, input_generators):
        for i, input in enumerate(estimator_inputs):

            if isinstance(input, np.ndarray):

                # Example: f_A or f_B of models with single output.
                # Shape: `(n_samples, 1)`.
                if input.ndim == 2 and input.shape[1] == 1:
                    input_generators.append(self.bootstrap_sample_generator_1D(input))

                # Example: f_C_i or f_D_i of models with single output.
                # Shape: `(n_samples, num_vars)`.
                elif input.ndim == 2 and input.shape[1] > 1:
                    input_generators.append(self.bootstrap_sample_generator_2D(input))

                # Example: f_C_i or f_D_i of models with multiple outputs.
                # Shape: `(n_outputs, n_samples, num_vars)`.
                elif input.ndim == 3:
                    input_generators.append(self.bootstrap_sample_generator_3D(input))

            elif input is None:
                input_generators.append(input)

            else:
                raise ValueError(f"UQpy: estimator_inputs[{i}] should be either "
                                 f"None or `ndarray` of dimension 1, 2 or 3")
