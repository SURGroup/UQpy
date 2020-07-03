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
``RunModel`` is the core module for ``UQpy`` to execute computational models

``RunModel`` contains a single class, also called ``RunModel`` that is used to execute computational models at specified
sample points. ``RunModel`` may be used to execute Python models or third-party software models and is capable of
running models serially or in parallel on both local machines or HPC clusters.

The module currently contains the following classes:

* ``RunModel``: Class for execution of a computational model

"""

import collections
import datetime
import glob
import os
import pathlib
import platform
import re
import shutil
import subprocess

import numpy as np


class RunModel:
    """
    Run a computational model at specified sample points.

    This class is the interface between ``UQpy`` and computational models. The model is called in a Python script whose
    name must be passed as one the arguments to the ``RunModel`` call. If the model is in Python, ``UQpy`` import the
    model and executes it directly. If the model is not in Python, ``RunModel`` must be provided the name of a template
    input file, the name of the Python script that runs the model, and an (optional) output Python script.

    **Input:**

    * **samples** (`ndarray` or `list`)
        Samples to be passed as inputs to the model.

        Regardless of data type, the first dimension of ``samples`` must be equal to the number of samples at which
        to execute the model. That is, ``len(samples) = nsamples``.

        Regardless of data type, the second dimension of ``samples`` must be equal to the number of variables to
        to pass for each model evaluation. That is, ``len(samples[0]) = n_vars``. Each variable need not be a scalar.
        Variables may be scalar, vector, matrix, or tensor type (i.e. `float`, `list`, or `ndarray`).

        If `samples` are not passed, a ``RunModel`` object will be instantiated that can be used later, with the ``run``
        method, to evaluate the model.

        Used in both python and third-party model execution.

    * **model_script** (`str`)
        The filename (with .py extension) of the Python script which contains commands to execute the model.

        The named file must be present in the current working directory from which ``RunModel`` is called.

    * **model_object_name** ('str`)
        In the Python workflow, `model_object_name` specifies the name of the function or class within `model_script'
        that executes the model. If there is only one function or class in the `model_script`, then it is not necessary
        to specify the model_object_name. If there are multiple objects within the `model_script`, then
        `model_object_name` must be specified.

        `model_object_name` is not used in the third-party software model workflow.

    * **input_template** (`str`)
        The name of the template input file that will be used to generate input files for each run of the model. When
        operating ``RunModel`` with a third-party software model, ``input_template`` must be specified.

        The named file must be present in the current working directory from which ``RunModel`` is called.

        `input_template` is not used in the Python model workflow.

    * **var_names** (`list` of `str`)
        A list containing the names of the variables present in `input_template`.

        If `input template` is provided and  `var_names` is not passed, i.e. if ``var_names=None``, then the default
        variable names `x0`, `x1`, `x2`,..., `xn` are created and used by ``RunModel``, where `n` is the number of
        variables (`n_vars`).

        The number of variables is equal to the second dimension of `samples` (i.e. ``n_vars=len(samples[0])``).

        `var_names` is not used in the Python model workflow.


    * **output_script** (`str`)
        The filename of the Python script that contains the commands to process the output from third-party software
        model evaluation. `output_script` is used to extract quantities of interest from model output files and return
        the quantities of interest to ``RunModel`` for subsequent ``UQpy`` processing (e.g. for adaptive methods that
        utilize the results of previous simulations to initialize new simulations).

        If, in the third-party software model workflow, ``output_script = None`` (the default), then the qoi_list
        attribute is empty and postprocessing must be handled outside of ``UQpy``.

        If used, the named file must be present in the current working directory from which ``RunModel`` is called.

        `output_script` is not used in the Python model workflow. In the Python model workflow, all model postprocessing
        is handled directly within `model_script`.

    * **output_object_name** (`str`)
        The name of the function or class within `output_script` that is used to collect and process the output values
        from third-party software model output files. If the object is a class, the output must be saved as an attribute
        called `qoi`. If it is a function, it should return the output quantity of interest.

        If there is only one function or only one class in `output_script`, then it is not necessary to specify
        `output_object_name`. If there are multiple objects in `output_script`, then output_object_name must be
        specified.

        `output_object_name` is not used in the Python model workflow.

    * **ntasks** (`int`)
        Number of tasks to be run in parallel. By default, ``ntasks = 1`` and the models are executed serially.

        Setting ntasks equal to a positive integer greater than 1 will trigger the parallel workflow.

        `ntasks` is used for both the Python and third-party model workflows. ``RunModel`` uses `GNU parallel` to
        execute third-party models in parallel and the multiprocessing module to execute Python models in parallel.

    * **cores_per_task** (`int`)
        Number of cores to be used by each task. In cases where a third-party model runs across multiple CPUs, this
        optional attribute allocates the necessary resources to each model evaluation.

        `cores_per_task` is not used in the Python model workflow.

    * **nodes** (`int`)
        Number of nodes across which to distribute individual tasks on an HPC cluster in the third-party model workflow.
        If more than one compute node is necessary to execute individual runs in parallel, `nodes` must be specified.

        `nodes` is not used in the Python model workflow.

    * **cluster** (`boolean`)
        Set ``cluster = True`` to run on an HPC cluster.

        ``RunModel`` currently supports computations on HPC clusters using the SLURM scheduler
        (https://slurm.schedmd.com). The set of model evaulations is submitted using the GNU `parallel` command with
        option `-j ntasks`. Individual model evaluations are submitted using the `srun` command with options `-N nodes`
        and `-c cores_per_node`.

        When ``cluster = False``, the `srun` command is not used, but the `parallel` command still is.

        `cluster` is not used for the Python model workflow.

    * **resume** (`boolean`)
        If ``resume = True``, `GNU parallel` enables ``UQpy`` to resume execution of any model evaluations that failed
        to execute in the third-party software model workflow.

        To use this feature, execute the same call to ``RunModel`` that failed to complete but with ``resume = True``.
        The same set of samples must be passed to resume processing from the last successful execution of the model.

        `resume` is not used in the Python model workflow.

    * **verbose** (`boolean`)
        Set ``verbose = True`` to print status messages to the terminal during execution.

    * **model_dir** (`str`)
        Specifies the name of the sub-directory from which the model will be executed and to which output files will be
        saved.  A new directory is created by ``RunModel`` within the current directory whose name is `model_dir`
        appended with a timestamp.

    * **fmt** (`str`)
        If the `template_input` requires variables to be written in specific format, this format can be specified here.

        Format specification follows standard Python conventions for the str.format() command described at:
        https://docs.python.org/3/library/stdtypes.html#str.format. For additional details, see the Format String Syntax
        description at: https://docs.python.org/3/library/string.html#formatstrings.

        For example, ls-dyna .k files require each card is to be exactly 10 characters. The following format string
        syntax can be used, "{:>10.4f}".

        `fmt` is not used in the Python model workflow.

    * **separator** (`str`)
        A string used to delimit values when printing arrays to the `template_input`.

        `separator` is not used in the Python model workflow.

    * **vec** (`boolean`)
        Specifies vectorized (``vec = True``) or looped (``vec = False``) model evaluation in the serial Python model
        workflow.

        In the Python model workflow, `model_script` may be written to accept a single sample or multiple samples at a
        time. If it is written to accept a single sample, set ``vec = False`` and ``RunModel`` will run the model in a
        loop over the number of samples. If `model_script` is written to accept multiple samples, set ``vec = True`` and
        ``RunModel`` will pass all of the samples to the model for vectorized computation.

        `vec` is not used in the third-party model workflow.

    * **delete_files** (`boolean`)
        Specifies whether or not to delete individual run output files after model execution and output processing.

        If `delete_files = True`, ``RunModel`` will remove all `run_i...` directories in the `model_dir`.

    * **kwargs** (`dict`)
        Additional inputs to the Python object specified by `model_object_name` in the Python model workflow.

        `**kwargs` is not used in the third-party model workflow.


    **Attributes**

    * **samples** (`ndarray`)
        Internally, ``RunModel`` converts the input `samples` into a numpy `ndarray` with at least two dimension where
        the first dimension of the `ndarray` corresponds to a single sample to be executed by the model.

    * **nsim** (`int`)
        Number of model evaluations to be performed, ``nsim = len(samples)``.

    * **nexist** (`int`)
        Number of pre-existing model evaluations, prior to a new ``run`` method call.

        If the ``run`` methods has previously been called and model evaluations performed, subsequent calls to the
        ``run`` method will be appended to the ``RunModel`` object. `nexist` stores the number of previously existing
        model evaluations.

    * **n_vars** (`int`)
        Number of variables to be passed for each model evaluation, ``n_vars = len(samples[0])``.

        Note that variables do not need to be scalars. Variables can be scalars, vectors, matrices, or tensors. When
        writing vectors, matrices, and tensors to a `input_template` they are first flattened and written in delimited
        form.

    * **qoi_list** (`list`)
        A list containing the output quantities of interest

        In the third-party model workflow, these output quantities of interest are extracted from the model output files
        by `output_script`.

        In the Python model workflow, the returned quantity of interest from the model evaluations is stored as
        `qoi_list`.

        This attribute is commonly used for adaptive algorithms that employ learning functions based on previous model
        evaluations.

    **Methods**
    """

    # Authors:
    # B.S. Aakash, Lohit Vandanapu, Michael D.Shields
    #
    # Last
    # modified: 5 / 8 / 2020 by Michael D.Shields

    def __init__(self, samples=None, model_script=None, model_object_name=None,
                 input_template=None, var_names=None, output_script=None, output_object_name=None, ntasks=1,
                 cores_per_task=1, nodes=1, cluster=False, resume=False, verbose=False, model_dir='Model_Runs',
                 fmt=None, separator=', ', vec=True, delete_files=False, **kwargs):

        # Check the platform and build appropriate call to Python
        if platform.system() in ['Windows']:
            self.python_command = "python"
        elif platform.system() in ['Darwin', 'Linux', 'Unix']:
            self.python_command = "python3"
        else:
            self.python_command = "python3"

        # Verbose option
        self.verbose = verbose

        # Vectorized computation
        self.vec = vec

        # Format option
        self.separator = separator
        self.fmt = fmt
        if self.fmt is None:
            pass
        elif isinstance(self.fmt, str):
            if (self.fmt[0] != "{") or (self.fmt[-1] != "}") or (":" not in self.fmt):
                raise ValueError('\nUQpy: fmt should be a string in brackets indicating a standard Python format.\n')
        else:
            raise TypeError('\nUQpy: fmt should be a str.\n')

        self.delete_files = delete_files

        # kwargs options, used only for python runs
        self.python_kwargs = kwargs

        # Input related
        self.input_template = input_template
        self.var_names = var_names
        self.n_vars = 0

        # Check if var_names is a list of strings
        if self.var_names is not None:
            if not self._is_list_of_strings(self.var_names):
                raise ValueError("\nUQpy: Variable names should be passed as a list of strings.\n")

        # Establish parent directory for simulations
        self.parent_dir = os.getcwd()

        # Create a list of all of the files and directories in the working directory. Do not include any other
        # directories containing the same name as model_dir
        model_files = []
        for f_name in os.listdir(self.parent_dir):
            path = os.path.join(self.parent_dir, f_name)
            if model_dir not in path:
                model_files.append(path)
        self.model_files = model_files

        # Create a new directory where the model will be executed
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")
        self.model_dir = os.path.join(self.parent_dir, model_dir + "_" + ts)
        os.makedirs(self.model_dir)
        if self.verbose:
            print('\nUQpy: The following directory has been created for model evaluations: \n' + self.model_dir)

        # Copy files from the model list to model run directory
        for file_name in model_files:
            full_file_name = os.path.join(self.parent_dir, file_name)
            if not os.path.isdir(full_file_name):
                shutil.copy(full_file_name, self.model_dir)
            else:
                new_dir_name = os.path.join(self.model_dir, os.path.basename(full_file_name))
                shutil.copytree(full_file_name, new_dir_name)
        if self.verbose:
            print('\nUQpy: The model files have been copied to the following directory for evaluation: \n' +
                  self.model_dir)

        # Check if the model script is a python script
        model_extension = pathlib.Path(model_script).suffix
        if model_extension == '.py':
            self.model_script = model_script
        else:
            raise ValueError("\nUQpy: The model script must be the name of a python script, with extension '.py'.")
        # Save the model object name
        self.model_object_name = model_object_name
        # Save option for resuming parallel execution
        self.resume = resume

        # Output related
        self.output_script = output_script
        self.output_object_name = output_object_name

        # Number of tasks
        self.ntasks = ntasks
        # Number of cores_per_task
        self.cores_per_task = cores_per_task
        # Number of nodes
        self.nodes = nodes
        self.template_text = ''
        self.output_module = None
        self.python_model = None

        # If running on cluster or not
        self.cluster = cluster

        # Initialize sample related variables
        self.samples = []
        self.samples = np.atleast_2d(self.samples)
        self.qoi_list = []
        self.nexist = 0
        self.nsim = 0

        # Check if samples are provided.
        if samples is None:
            if self.verbose:
                print("\nUQpy: No samples are provided. Creating the object and building the model directory.\n")
        elif isinstance(samples, (list, np.ndarray)):
            self.run(samples)
        else:
            raise ValueError("\nUQpy: samples must be passed as a list or numpy ndarray\n")

    def run(self, samples=None, append_samples=True):
        """
        Execute a computational model at given sample values.

        If `samples` are passed when defining the ``RunModel`` object, the ``run`` method is called automatically.

        The ``run`` method may also be called directly after defining the ``RunModel`` object.

        **Input:**

        * **samples** (`ndarray` or `list`)
            Samples to be passed as inputs to the model defined by the ``RunModel`` object.

            Regardless of data type, the first dimension of ``samples`` must be equal to the number of samples at which
            to execute the model. That is, ``len(samples) = nsamples``.

            Regardless of data type, the second dimension of ``samples`` must be equal to the number of variables to
            to pass for each model evaluation. That is, ``len(samples[0]) = n_vars``. Each variable need not be a
            scalar. Variables may be scalar, vector, matrix, or tensor type (i.e. `float`, `list`, or `ndarray`).

            Used in both python and third-party model execution.

        * **append_samples** (`boolean`)
            Append over overwrite existing samples and model evaluations.

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

        # Change current working directory to model run directory
        os.chdir(self.model_dir)
        if self.verbose:
            print('\nUQpy: All model evaluations will be executed from the following directory: \n' + self.model_dir)

        # Number of simulations to be performed
        self.nsim = len(samples)

        # Number of variables
        self.n_vars = len(samples[0])

        # If append_samples is False, a new set of samples is created, the previous ones are deleted!
        if not append_samples:
            self.samples = []
            self.samples = np.atleast_2d(self.samples)
            self.qoi_list = []

        # Check if samples already exist, if yes append new samples to old ones
        # if not self.samples:  # There are currently no samples
        if self.samples.size == 0:

            # If there are no samples, check to ensure that len(var_names) = n_vars
            if self.input_template is not None:
                if self.var_names is not None:
                    # Check to see if self.var_names has the correct length
                    if len(self.var_names) != self.n_vars:
                        raise ValueError("\nUQpy: var_names must have the same length as the number of variables (i.e. "
                                         "len(var_names) = len(samples[0]).\n")
                else:
                    # If var_names is not passed and there is an input template, create default variable names
                    self.var_names = []
                    for i in range(self.n_vars):
                        self.var_names.append('x%d' % i)

            self.nexist = 0
            self.samples = samples
            self.qoi_list = [None] * self.nsim

        else:  # Samples already exist in the RunModel object, append new ones
            self.nexist = len(self.samples)
            self.qoi_list.extend([None] * self.nsim)
            self.samples = np.vstack((self.samples, samples))

        # Check if there is a template input file or not and execute the appropriate function
        if self.input_template is not None:  # If there is a template input file
            # Check if it is a file and is readable
            assert os.path.isfile(self.input_template) and os.access(self.input_template, os.R_OK), \
                "\nUQpy: File {} doesn't exist or isn't readable".format(self.input_template)
            # Read in the text from the template files
            with open(self.input_template, 'r') as f:
                self.template_text = str(f.read())

            # Import the output script
            if self.output_script is not None:
                self.output_module = __import__(self.output_script[:-3])
                # Run function which checks if the output module has the output object
                self._check_output_module()

            # Run the serial execution or parallel execution depending on ntasks
            if self.ntasks == 1:
                self._serial_execution()
            else:
                self._parallel_execution()

        else:  # If there is no template input file supplied
            # Import the python module
            self.python_model = __import__(self.model_script[:-3])
            # Run function which checks if the python model has the model object
            self._check_python_model()

            # Run the serial execution or parallel execution depending on ntasks
            if self.ntasks == 1:
                self._serial_python_execution()
            else:
                self._parallel_python_execution()

        # Return to parent directory
        if self.verbose:
            print("\nUQpy: Returning to the parent directory:\n" + self.parent_dir)
        os.chdir(self.parent_dir)

        if self.delete_files:
            if self.verbose:
                print("UQpy: Deleting individual run files.")
            for dirname in glob.glob(os.path.join(self.model_dir, "run*")):
                shutil.rmtree(dirname)

        return None

    ####################################################################################################################

    def _serial_execution(self):
        """
        Perform serial execution of a third-party model using a template input file

        This function loops over the number of simulations, executing the model once per loop. In each loop, the
        function creates a directory for each model run, copies files to the model run directory,
        changes the current working directory to the model run directory, calls the input function, executes the model,
        calls the output function, removes the copied files and folders, and returns to the previous directory.
        """
        if self.verbose:
            print('\nUQpy: Performing serial execution of the third-party model.\n')

        # Loop over the number of simulations, executing the model once per loop
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")
        for i in range(self.nexist, self.nexist + self.nsim):
            # Create a directory for each model run
            work_dir = os.path.join(self.model_dir, "run_" + str(i) + '_' + ts)
            self._copy_files(work_dir=work_dir)

            # Change current working directory to model run directory
            os.chdir(work_dir)
            if self.verbose:
                print('\nUQpy: Running model number ' + str(i) + ' in the following directory: \n' + work_dir)

            # Call the input function
            self._input_serial(i)

            # Execute the model
            self._execute_serial(i)

            # Call the output function
            if self.output_script is not None:
                self._output_serial(i)

            # Remove the copied files and folders
            self._remove_copied_files(work_dir)

            # Return to the model directory
            os.chdir(self.model_dir)
            if self.verbose:
                print('\nUQpy: Model evaluation ' + str(i) + ' complete.\n')
                print('\nUQpy: Returning to the model directory:\n' + self.model_dir)

        if self.verbose:
            print('\nUQpy: Serial execution of the third-party model complete.\n')

    ####################################################################################################################

    def _parallel_execution(self):
        """
        Execute a third-party model in parallel

        This function calls the input function and generates input files for all the samples, then creates a directory
        for each model run, copies files to the model run directory, executes the model in parallel, collects output,
        removes the copied files and folders.
        """
        if self.verbose:
            print('\nUQpy: Performing parallel execution of the third-party model.\n')
            # Call the input function
            print('\nUQpy: Creating inputs for parallel execution of the third-party model.\n')

        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")

        # Create all input files for the parallel execution and place them in the proper directories
        for i in range(self.nexist, self.nexist + self.nsim):
            # Create a directory for each model run
            work_dir = os.path.join(self.model_dir, "run_" + str(i) + '_' + ts)
            self._copy_files(work_dir=work_dir)

        self._input_parallel(ts)

        # Execute the model
        if self.verbose:
            print('\nUQpy: Executing the third-party model in parallel.\n')

        self._execute_parallel(ts)

        # Call the output function
        if self.verbose:
            print('\nUQpy: Collecting outputs from parallel execution of the third-party model.\n')

        for i in range(self.nexist, self.nexist + self.nsim):
            # Change current working directory to model run directory
            work_dir = os.path.join(self.model_dir, "run_" + str(i) + '_' + ts)
            if self.verbose:
                print('\nUQpy: Changing to the following directory for output processing:\n' + work_dir)
            os.chdir(work_dir)

            # Run output processing function
            if self.output_script is not None:
                if self.verbose:
                    print('\nUQpy: Processing output from parallel execution of the third-party model run ' + str(i) +
                          '.\n')
                self._output_serial(i)

            # Remove the copied files and folders
            self._remove_copied_files(work_dir)

            # Change back to the upper directory
            if self.verbose:
                print('\nUQpy: Changing back to the following model directory:\n' + self.model_dir)
            os.chdir(self.model_dir)

        if self.verbose:
            print('\nUQpy: Parallel execution of the third-party model complete.\n')

    ####################################################################################################################
    def _serial_python_execution(self):
        """
        Execute a python model in serial

        This function imports the model_object from the model_script, and executes the model in series by passing the
        corresponding sample/samples along with keyword arguments, if any, as inputs to the model object.
        """
        if self.verbose:
            print('\nUQpy: Performing serial execution of a Python model.\n')

        model_object = getattr(self.python_model, self.model_object_name)
        # Run python model
        if self.vec:
            # If the Python model is vectorized to accept many samples.
            self.model_output = model_object(self.samples[self.nexist:self.nexist + self.nsim], **self.python_kwargs)
            if self.model_is_class:
                self.qoi_list[self.nexist:self.nexist+self.nsim] = list(self.model_output.qoi)
            else:
                self.qoi_list[self.nexist:self.nexist+self.nsim] = list(self.model_output)
        else:
            # If the Python model is not vectorized and accepts only a single sample.
            for i in range(self.nexist, self.nexist + self.nsim):
                sample_to_send = np.atleast_2d(self.samples[i])

                if len(self.python_kwargs) == 0:
                    self.model_output = model_object(sample_to_send)
                else:
                    self.model_output = model_object(sample_to_send, **self.python_kwargs)
                if self.model_is_class:
                    self.qoi_list[i] = self.model_output.qoi
                else:
                    self.qoi_list[i] = self.model_output

        if self.verbose:
            print('\nUQpy: Serial execution of the python model complete.\n')

        if self.verbose:
            print('\nUQpy: Serial execution of the python model complete.\n')

    ####################################################################################################################
    def _parallel_python_execution(self):
        """
        Execute a python model in parallel

        This function imports the model object from the model script, and executes the model in parallel by passing the
        samples along with keyword arguments, if any, as inputs to the model object.
        """

        if self.verbose:
            print('\nUQpy: Performing parallel execution of the model without template input.\n')
        import multiprocessing
        import UQpy.Utilities as Utilities

        sample = []
        pool = multiprocessing.Pool(processes=self.ntasks)
        for i in range(self.nexist, self.nexist + self.nsim):
            sample_to_send = np.atleast_2d(self.samples[i])
            if len(self.python_kwargs) == 0:
                sample.append([self.model_script, self.model_object_name, sample_to_send])
            else:
                sample.append([self.model_script, self.model_object_name, sample_to_send, self.python_kwargs])

        results = pool.starmap(Utilities.run_parallel_python, sample)

        for i in range(self.nsim):
            if self.model_is_class:
                self.qoi_list[i + self.nexist] = results[i].qoi
            else:
                self.qoi_list[i + self.nexist] = results[i]

        pool.close()

        if self.verbose:
            print('\nUQpy: Parallel execution of the python model complete.\n')

    ####################################################################################################################
    def _input_serial(self, index):
        """
        Create one input file using the template and attach the index to the filename

        ** Input: **

        :param index: The simulation number
        :type index: int
        """
        self.new_text = self._find_and_replace_var_names_with_values(index=index)
        # Write the new text to the input file
        self._create_input_files(file_name=self.input_template, num=index, text=self.new_text, new_folder='InputFiles')

    def _execute_serial(self, index):
        """
        Execute the model once using the input file of index number

        ** Input: **

        :param index: The simulation number
        :type index: int
        """
        self.model_command = ([self.python_command, str(self.model_script), str(index)])
        subprocess.run(self.model_command)

    def _output_serial(self, index):
        """
        Execute the output script, obtain the output qoi and save it in qoi_list

        ** Input: **

        :param index: The simulation number
        :type index: int
        """
        # Run output module
        self.output_module = __import__(self.output_script[:-3])
        output_object = getattr(self.output_module, self.output_object_name)
        self.model_output = output_object(index)

        if self.output_is_class:
            self.qoi_list[index] = self.model_output.qoi
        else:
            self.qoi_list[index] = self.model_output

    def _input_parallel(self, timestamp):
        """
        Create all the input files required

        ** Input: **

        :param timestamp: Timestamp which is appended to the name of the input file
        :type timestamp: str
        """
        # Loop over the number of samples and create input files in a folder in current directory
        for i in range(self.nsim):
            new_text = self._find_and_replace_var_names_with_values(index=i + self.nexist)
            folder_to_write = 'run_' + str(i+self.nexist) + '_' + timestamp + '/InputFiles'
            # Write the new text to the input file
            self._create_input_files(file_name=self.input_template, num=i+self.nexist, text=new_text,
                                     new_folder=folder_to_write)
            if self.verbose:
                print('\nUQpy: Created input files for run ' + str(i) + ' in the directory: \n' +
                      os.path.join(self.model_dir, folder_to_write))

    def _execute_parallel(self, timestamp):
        """
        Build the command string and execute the model in parallel using subprocess and gnu parallel

        ** Input: **

        :param timestamp: Timestamp which is appended to the name of the input file
        :type timestamp: str
        """

        # TODO: Generalize run_string and parallel_string for any cluster
        # Check if logs folder exists, if not, create it
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # If the user sets resume=True, do not delete log file. Else, delete logfile before running
        if self.resume is False:
            try:
                os.remove("logs/runtask.log")
            except OSError:
                pass
        self.parallel_string = "parallel --delay 0.2 --joblog logs/runtask.log --resume -j " + str(self.ntasks) + " "

        # If running on SLURM cluster
        if self.cluster:
            self.srun_string = "srun -N" + str(self.nodes) + " -n1 -c" + str(self.cores_per_task) + " --exclusive "
            self.model_command_string = (self.parallel_string + "'(cd run_{1}_" + timestamp + " && " + self.srun_string
                                         + " " + self.python_command + " -u " + str(self.model_script) +
                                         " {1})'  ::: {" + str(self.nexist) + ".." +
                                         str(self.nexist + self.nsim - 1) + "}")
        else:  # If running locally
            self.model_command_string = (self.parallel_string + " 'cd run_{1}_" + timestamp + " && " +
                                         self.python_command + " -u " +
                                         str(self.model_script) + "' {1}  ::: {" + str(self.nexist) + ".." +
                                         str(self.nexist + self.nsim - 1) + "}")

        subprocess.run(self.model_command_string, shell=True)

    ####################################################################################################################
    # Helper functions

    @staticmethod
    def _create_input_files(file_name, num, text, new_folder='InputFiles'):
        """
        Create input files using filename, index, text

        ** Input: **

        :param file_name: Name of input file
        :type file_name: str

        :param num: The simulation number
        :type num: int

        :param text: Contents of the input file
        :type text: str

        :param new_folder: Name of directory where the created input files are placed

                           Default: 'InputFiles'
        :type new_folder: str
        """
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        base_name = os.path.splitext(os.path.basename(file_name))
        new_name = os.path.join(new_folder, base_name[0] + "_" + str(num) + base_name[1])
        with open(new_name, 'w') as f:
            f.write(text)
        return

    def _find_and_replace_var_names_with_values(self, index):
        """
        Replace placeholders containing variable names in template input text with sample values.

        ** Input: **

        :param index: The sample number
        :type index: int
        """

        template_text = self.template_text
        var_names = self.var_names
        samples = self.samples[index]

        new_text = template_text
        for j in range(self.n_vars):
            string_regex = re.compile(r"<" + var_names[j] + r".*?>")
            count = 0
            for string in string_regex.findall(template_text):
                temp_check = string[1:-1].split("[")[0]
                pattern_check = re.compile(var_names[j])
                if pattern_check.fullmatch(temp_check):
                    temp = string[1:-1].replace(var_names[j], "samples[" + str(j) + "]")
                    try:
                        temp = eval(temp)
                    except IndexError as err:
                        print("\nUQpy: Index Error: {0}\n".format(err))
                        raise IndexError("{0}".format(err))

                    if isinstance(temp, collections.Iterable):
                        # If it is iterable, flatten and write as text file with designated separator
                        temp = np.array(temp).flatten()
                        to_add = ''
                        for i in range(len(temp) - 1):
                            if self.fmt is None:
                                to_add += str(temp[i]) + self.separator
                            else:
                                to_add += self.fmt.format(temp[i]) + self.separator
                        if self.fmt is None:
                            to_add += str(temp[-1])
                        else:
                            to_add += self.fmt.format(temp[-1])
                    else:
                        if self.fmt is None:
                            to_add = str(temp)
                        else:
                            to_add = self.fmt.format(temp)
                    new_text = new_text[0:new_text.index(string)] + to_add + new_text[(new_text.index(string) +
                                                                                       len(string)):]
                    count += 1
            if self.verbose:
                if index == 0:
                    if count > 1:
                        print(
                            "\nUQpy: Found " + str(count) + " instances of variable: '" + var_names[j] +
                            "' in the input file.\n")
                    else:
                        print(
                            "\nUQpy: Found " + str(count) + " instance of variable: '" + var_names[j] +
                            "' in the input file.\n")
        return new_text

    def _remove_copied_files(self, work_dir):
        """
        Remove the copied files from each run directory to avoid having many redundant files.

        ** Input: **

        :param work_dir: The working directory of the current run.
        :type work_dir: str
        """

        for file_name in self.model_files:
            full_file_name = os.path.join(work_dir, os.path.basename(file_name))
            if not os.path.isdir(full_file_name):
                os.remove(full_file_name)
            else:
                shutil.rmtree(full_file_name)

    @staticmethod
    def _is_list_of_strings(list_of_strings):
        """
        Check if input list contains only strings

        ** Input: **

        :param list_of_strings: A list whose entries should be checked to see if they are strings
        :type list_of_strings: list
        """
        return bool(list_of_strings) and isinstance(list_of_strings, list) and all(isinstance(element, str) for element
                                                                                   in list_of_strings)

    def _check_python_model(self):
        """
        Check if python model name is valid

        This function gets the name of the classes and functions in the imported python module whose names is passed in
        as the python model to RunModel. There should be at least one class or function in the module - if not there,
        then the function exits raising a ValueError. If there is at least one class or function in the module,
        if the model object name is not given as input and there is only one class or function, that class name or
        function name is used to run the model. If there is a model_object_name given, check if it is a valid name.
        Else, a ValueError is raised.
        """
        # Get the names of the classes and functions in the imported module
        import inspect
        class_list = []
        function_list = []
        for name, obj in inspect.getmembers(self.python_model):
            if inspect.isclass(obj):
                class_list.append(name)
            elif inspect.isfunction(obj):
                function_list.append(name)

        # There should be at least one class or function in the module - if not there, exit with error.
        if class_list is [] and function_list is []:
            raise ValueError(
                "\nUQpy: A python model should be defined as a function or class in the script.\n")

        else:  # If there is at least one class or function in the module
            # If the model object name is not given as input and there is only one class or function,
            # take that class name or function name to run the model.
            if self.model_object_name is None and len(class_list) + len(function_list) == 1:
                if len(class_list) == 1:
                    self.model_object_name = class_list[0]
                elif len(function_list) == 1:
                    self.model_object_name = function_list[0]

            # If there is a model_object_name given, check if it is in the list.
            if self.model_object_name in class_list:
                if self.verbose:
                    print('\nUQpy: The model class that will be run: ' + self.model_object_name)
                self.model_is_class = True
            elif self.model_object_name in function_list:
                if self.verbose:
                    print('\nUQpy: The model function that will be run: ' + self.model_object_name)
                self.model_is_class = False
            else:
                if self.model_object_name is None:
                    raise ValueError("\nUQpy: There are more than one objects in the module. Specify the name of the "
                                     "function or class which has to be executed.\n")
                else:
                    print('\nUQpy: You specified the model_object_name as: ' + str(self.model_object_name))
                    raise ValueError("\nUQpy: The file does not contain an object which was specified as the model.\n")

    def _check_output_module(self):
        """
        Check if output script name is valid

        This function get the names of the classes and functions in the imported module. There should be at least one
        class or function in the module - if not there, exit with ValueError. If there is at least one class or
        function in the module, if the output object name is not given as input and there is only one class or function,
        take that class name or function name to extract output. If there is a output_object_name given, check if it is
        a valid name. Else, a ValueError is raised.
        """
        # Get the names of the classes and functions in the imported module
        import inspect
        class_list = []
        function_list = []
        for name, obj in inspect.getmembers(self.output_module):
            if inspect.isclass(obj):
                class_list.append(name)
            elif inspect.isfunction(obj):
                function_list.append(name)

        # There should be at least one class or function in the module - if not there, exit with error.
        if class_list is [] and function_list is []:
            raise ValueError(
                "\nUQpy: The output object should be defined as a function or class in the script.\n")

        else:  # If there is at least one class or function in the module
            # If the model object name is not given as input and there is only one class or function,
            # take that class name or function name to run the model.
            if self.output_object_name is None and len(class_list) + len(function_list) == 1:
                if len(class_list) == 1:
                    self.output_object_name = class_list[0]
                elif len(function_list) == 1:
                    self.output_object_name = function_list[0]

            # If there is a model_object_name given, check if it is in the list.
            if self.output_object_name in class_list:
                if self.verbose:
                    print('\nUQpy: The output class that will be run: ' + self.output_object_name)
                self.output_is_class = True
            elif self.output_object_name in function_list:
                if self.verbose:
                    print('\nUQpy: The output function that will be run: ' + self.output_object_name)
                self.output_is_class = False
            else:
                if self.output_object_name is None:
                    raise ValueError("\nUQpy: There are more than one objects in the module. Specify the name of the "
                                     "function or class which has to be executed.\n")
                else:
                    print('\nUQpy: You specified the output object name as: ' + str(self.output_object_name))
                    raise ValueError("\nUQpy: The file does not contain an object which was specified as the output "
                                     "processor.\n")

    def _copy_files(self, work_dir):
        os.makedirs(work_dir)

        # Copy files from the model list to model run directory
        for file_name in self.model_files:
            full_file_name = os.path.join(self.model_dir, file_name)
            if not os.path.isdir(full_file_name):
                shutil.copy(full_file_name, work_dir)
            else:
                new_dir_name = os.path.join(work_dir, os.path.basename(full_file_name))
                shutil.copytree(full_file_name, new_dir_name)
