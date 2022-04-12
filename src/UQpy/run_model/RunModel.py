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

import logging
import os
import pickle
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import NumpyFloatArray


class RunModel:
    # Authors:
    # B.S. Aakash, Lohit Vandanapu, Michael D.Shields
    #
    # Last
    # modified: 5 / 8 / 2020 by Michael D.Shields
    @beartype
    def __init__(
            self,
            model,
            samples: Union[list, NumpyFloatArray] = None,
            ntasks: int = 1,
            cores_per_task: int = 1,
            nodes: int = 1,
            resume: bool = False,
    ):
        """
        Run a computational model at specified sample points.

        This class is the interface between :py:mod:`UQpy` and computational models. The model is called in a Python script whose
        name must be passed as one the arguments to the :class:`.RunModel` call. If the model is in Python, :py:mod:`UQpy` import the
        model and executes it directly. If the model is not in Python, :class:`.RunModel` must be provided the name of a template
        input file, the name of the Python script that runs the model, and an (optional) output Python script.


        :param samples: Samples to be passed as inputs to the model.
         Regardless of data type, the first dimension of ``samples`` must be equal to the number of samples at which
         to execute the model. That is, ``len(samples) = nsamples``.

         Regardless of data type, the second dimension of ``samples`` must be equal to the number of variables to
         to pass for each model evaluation. That is, ``len(samples[0]) = n_vars``. Each variable need not be a scalar.
         Variables may be scalar, vector, matrix, or tensor type (i.e. `float`, `list`, or `ndarray`).

         If `samples` are not passed, a :class:`.RunModel` object will be instantiated that can be used later, with the
         :meth:`run` method, to evaluate the model.

         Used in both python and third-party model execution.

        :param ntasks: Number of tasks to be run in parallel. By default, ``ntasks = 1`` and the models are executed
         serially.

         Setting ntasks equal to a positive integer greater than 1 will trigger the parallel workflow.

         `ntasks` is used for both the Python and third-party model workflows. :class:`.RunModel` uses `GNU parallel` to
         execute third-party models in parallel and the multiprocessing module to execute Python models in parallel.

        :param cores_per_task: Number of cores to be used by each task. In cases where a third-party model runs across
         multiple CPUs, this optional attribute allocates the necessary resources to each model evaluation.

         `cores_per_task` is not used in the Python model workflow.

        :param nodes: Number of nodes across which to distribute individual tasks on an HPC cluster in the third-party
         model workflow. If more than one compute node is necessary to execute individual runs in parallel, `nodes` must
         be specified.
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        # Save option for resuming parallel execution
        self.resume = resume

        self.nodes = nodes
        self.ntasks = ntasks
        self.cores_per_task = cores_per_task

        self.is_serial = ntasks <= 1 and cores_per_task <= 1 and nodes <= 1

        # Initialize sample related variables
        self.samples: NumpyFloatArray = []
        """Internally, :class:`.RunModel` converts the input `samples` into a numpy `ndarray` with at least two 
        dimension where the first dimension of the :class:`numpy.ndarray` corresponds to a single sample to be executed 
        by the model."""
        self.samples = np.atleast_2d(self.samples)
        self.qoi_list: list = []
        """A list containing the output quantities of interest

        In the third-party model workflow, these output quantities of interest are extracted from the model output files
        by `output_script`.

        In the Python model workflow, the returned quantity of interest from the model evaluations is stored as
        :py:attr:`qoi_list`.

        This attribute is commonly used for adaptive algorithms that employ learning functions based on previous model
        evaluations."""
        self.n_existing_simulations: int = 0
        """Number of pre-existing model evaluations, prior to a new :meth:`run` method call.

        If the :meth:`run` methods has previously been called and model evaluations performed, subsequent calls to the
        :meth:`run` method will be appended to the :class:`RunModel` object. :py:attr:`nexist` stores the number of 
        previously existing model evaluations."""
        self.n_new_simulations: int = 0
        """Number of model evaluations to be performed, ``nsim = len(samples)``."""

        # Check if samples are provided.
        if samples is None:
            self.logger.info("\nUQpy: No samples are provided. Creating the object and building the model directory.\n")
        elif isinstance(samples, (list, np.ndarray)):
            self.run(samples)
        else:
            raise ValueError("\nUQpy: samples must be passed as a list or numpy ndarray\n")

    def run(self, samples=None, append_samples=True):
        """
        Execute a computational model at given sample values.

        If `samples` are passed when defining the :class:`.RunModel` object, the :meth:`run` method is called automatically.

        The :meth:`run` method may also be called directly after defining the :meth:`RunModel` object.

        :param list samples: Samples to be passed as inputs to the model defined by the :class:`.RunModel` object.

         Regardless of data type, the first dimension of ``samples`` must be equal to the number of samples at which
         to execute the model. That is, ``len(samples) = nsamples``.

         Regardless of data type, the second dimension of ``samples`` must be equal to the number of variables to
         to pass for each model evaluation. That is, ``len(samples[0]) = n_vars``. Each variable need not be a
         scalar. Variables may be scalar, vector, matrix, or tensor type (i.e. `float`, `list`, or `ndarray`).

         Used in both python and third-party model execution.
        :param bool append_samples: Append over overwrite existing samples and model evaluations.

         If ``append_samples = False``, all previous samples and the corresponding quantities of interest from their
         model evaluations are deleted.

         If ``append_samples = True``, samples and their resulting quantities of interest are appended to the
         existing ones.
        """
        # Ensure the input samples have the correct structure
        # --> If a list is provided, convert to at least 2d ndarray. dim1 = nsim, dim2 = n_vars
        # --> If 1D array/list is provided, convert it to a 2d array. dim1 = 1, dim2 = n_vars
        # --> If samples cannot be converted to an array, this will fail.
        samples = np.atleast_2d(samples)

        # Number of simulations to be performed
        self.n_new_simulations = len(samples)

        # If append_samples is False, a new set of samples is created, the previous ones are deleted!
        if not append_samples:
            self.samples = []
            self.samples = np.atleast_2d(self.samples)
            self.qoi_list = []

        # Check if samples already exist, if yes append new samples to old ones
        # if not self.samples:  # There are currently no samples
        if self.samples.size == 0:
            self.n_existing_simulations = 0
            self.samples = samples

        else:  # Samples already exist in the RunModel object, append new ones
            self.n_existing_simulations = len(self.samples)
            self.samples = np.vstack((self.samples, samples))

        self.model.initialize(samples)

        self.qoi_list.extend(self.serial_execution() if self.is_serial else self.parallel_execution())

        self.model.finalize()

    def parallel_execution(self):
        # TODO: Check if files with the names used below already exist and raise error
        with open('model.pkl', 'wb') as filehandle:
            pickle.dump(self.model, filehandle)
        with open('samples.pkl', 'wb') as filehandle:
            pickle.dump(self.samples, filehandle)
        os.system(f"mpirun python -m "
                  f"UQpy.run_model.model_execution.ParallelExecution {self.n_existing_simulations} "
                  f"{self.n_new_simulations}")
        with open('qoi.pkl', 'rb') as filehandle:
            results = pickle.load(filehandle)

        os.remove("model.pkl")
        os.remove("samples.pkl")
        os.remove("qoi.pkl")

        self.logger.info("\nUQpy: Parallel execution of the python model complete.\n")
        return results

    def serial_execution(self):
        results = []
        for i in range(self.n_existing_simulations, self.n_existing_simulations + self.n_new_simulations):
            sample = self.model.preprocess_single_sample(i, self.samples[i])

            execution_output = self.model.execute_single_sample(i, sample)

            results.append(self.model.postprocess_single_file(i, execution_output))

        self.logger.info("\nUQpy: Serial execution of the python model complete.\n")
        return results
