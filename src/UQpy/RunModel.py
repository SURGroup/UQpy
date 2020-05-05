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
RunModel is the core module for UQpy to execute computational models

RunModel contains a single class, also called RunModel that is used to execute computational models at specified sample
points. RunModel may be used to execute Python models or third-party software models and is capable of running models
serially or in parallel on both local machines or HPC clusters.

The module currently contains the following classes:

* RunModel - Class for execution of a computational model
"""

import os
import subprocess
import pathlib
import re
import collections
import numpy as np
import datetime
import shutil
import platform


class RunModel:
    """
    Run a computational model at specified sample points.

    This class is the interface between UQpy and computational models. The model is called in a Python script whose
    name must be passed as one the arguments to the RunModel call. If the model is in Python, UQpy can interface with
    the model without the need for an input file. In this case, UQpy imports the model module and executes the model
    object. If the model is not in Python, RunModel must be provided the name of a template input file and an output
    Python script along with the name of the Python script that runs the model.

    **Input:**

    :param samples: Samples to be passed as inputs to the model. Samples can be passed either as an ndarray or a list.
                    If an ndarray is passed, the first dimension of the array must correspond to the number of samples
                    at which to execute the model and the second dimension must correspond to the number of variables,
                    or dimension . If a list is passed, each item of the list contains one set of samples required for
                    one execution of the model. In either case, number of samples = len(samples) and
                    dimension = len(samples[0]). If this is not the case, RunModel will not accept the samples.
    :type samples: ndarray or list

    :param model_script: The filename (with extension) of the Python script which contains commands to execute the
                         model. The model script must be present in the current working directory from which RunModel is
                         called.
    :type model_script: str

    :param model_object_name: In the Python model workflow, model_object_name specifies the name of the function or
                              class within model_script which executes the model. If there is only one function or class
                              in the model_script, then it is not necessary to specify the model_object_name. If there
                              are multiple objects within the model_script, then model_object_name must be specified.
                              model_object_name is not used in the third-party software model workflow.
    :type model_object_name: str

    :param input_template: The name of the template input file which will be used to generate input files for each
                           run of the model. When operating RunModel with a third-party software model, input_template
                           must be specified. input_template is not used in the Python model workflow.
    :type input_template: str

    :param var_names: A list containing the names of the variables present in the template input file. If an
                      input template is provided and a list of variable names is not passed, ie if var_names=None, then
                      the default variable names x0, x1, x2,...,xn are created and used by RunModel, where n is the
                      number of variables. The number of variables is equal to the shape of the first row if samples
                      is passed as an ndarray or the shape of the first item if samples is passed as a list. var_names
                      is not used in the Python model workflow.
    :type var_names: list of str or None

    :param output_script: The filename of the Python script that contains the commands to process the output from
                          third-party software model evaluation. The output_script is used to return the output
                          quantities of interest to RunModel for subsequent UQpy processing (e.g. for adaptive methods
                          that utilize the results of previous simulations to initialize new simulations). output_script
                          is not used in the Python model workflow. In the Python model workflow, all model
                          postprocessing is handled within model_script. If, in the third-party software model workflow,
                          output_script = None (the default), then RunModel.qoi_list is empty and postprocessing must be
                          handled outside of UQpy.
    :type output_script: str

    :param output_object_name: The name of the function or class that is used to collect and process the output values
                               from third-party software model output files. If the object is a class named cls, for
                               example, the output must be saved as cls.qoi. If it is a function, it should
                               return the output quantity of interest. If there is only one function or only one class
                               in output_script, then it is not necessary to specify output_object_name. If there are
                               multiple objects in output_script, then output_object_name must be specified.
                               output_object_name is not used in the Python model workflow.
    :type output_object_name: str

    :param ntasks: Number of tasks to be run in parallel. By default, ntasks = 1 and the models are executed serially.
                   Setting ntasks equal to a positive integer greater than 1 will trigger the parallel workflow.
                   RunModel uses GNU parallel to execute models which require an input template in parallel and the
                   concurrent module to execute Python models in parallel.
    :type ntasks: int

    :param cores_per_task: Number of cores to be used by each task. In cases where a third-party model runs across
                           multiple CPUs, this optional attribute allocates the necessary resources to each model
                           evaluation. cores_per_task is not used in the Python model workflow.
    :type cores_per_task: int

    :param nodes: Number of nodes across which to distribute parallel jobs on an HPC cluster in the third-party software
                  model workflow. If more than one compute node is required to execute the runs in parallel, nodes must
                  be specified. For example, on the Maryland Advanced Research Computing Center (MARCC), an HPC shared
                  by Johns Hopkins University and the University of Maryland, each compute node has 24 cores. To run an
                  analysis with more than 24 parallel jobs on MARCC requires nodes > 1. nodes is not used in the Python
                  model workflow.
    :type nodes: int

    :param resume: If resume = True, GNU parallel enables UQpy to resume execution of any model evaluations that failed
                   to execute in the third-party software model workflow. To use this feature, execute the same call to
                   RunModel which failed to complete but with resume = True.  The same set of samples must be passed to
                   resume processing from the last successful execution of the model. resume is not used in the Python
                   model workflow.
    :type resume: Boolean

    :param verbose: Set verbose = True if you want RunModel to print status messages to the terminal during execution.
                    verbose = False by default.
    :type verbose: Boolean

    :param model_dir: Specifies the name of the sub-directory from which the model will be executed and to which output
                      files will be saved. model_dir = None by default, which results in model execution from the Python
                      current working directory. If model_dir is passed a string, then a new directory is created by
                      RunModel within the current directory whose name is model_dir appended with a timestamp.
    :type model_dir: str

    :param cluster: Set cluster = True if executing on an HPC cluster. Setting cluster = True enables RunModel to
                    execute the model using the necessary SLURM commands. cluster = False by default. RunModel is
                    configured for HPC clusters that operate with the SLURM scheduler. In order to execute a third-party
                    model with RunModel on an HPC cluster, the HPC must use SLURM. cluster is not used for the Python
                    model workflow.
    :type cluster: Boolean

    :param fmt: If the input file requires variables to be written in specific format, this format can be specified
                here. Format specification follows standard Python conventions for the str.format() command described
                at: https://docs.python.org/3/library/stdtypes.html#str.format For additional details, see the Format
                String Syntax description at: https://docs.python.org/3/library/string.html#formatstrings The default is
                fmt = None. Some noteworthy formats include:

                * For ls-dyna .k files, each card is required to be exactly 10 characters. The following format string
                  syntax is recommended, "{:>10.4f}".
    :type fmt: str

    :param kwargs: Additional inputs to the python function (model_object_name). This option is only used for execution
                   of Python models.
    :type kwargs: dictionary

    :param separator: A string used to delimit values when printing arrays to input files. Default = ', '

    type separator: str

    **Attributes**

    :param RunModel.qoi_list: A list containing the output quantities of interest extracted from the model output files
                              by output_script. This is a list of length equal to the number of simulations. Each item
                              of this list contains the quantity of interest from the associated simulation.
    :type RunModel.qoi_list: list

    :param RunModel.samples: Internally, RunModel converts samples into a numpy array with at least two dimension where
                             each row (first dimension of the array) of the list corresponds to a single sample to be
                             executed by the model is a ndarray.
    :type RunModel.samples: ndarray

    **Authors:**

    B.S. Aakash, Lohit Vandanapu, Michael D. Shields

    Last modified: 3/26/2020 by B.S. Aakash
    """

    def __init__(self, samples=None, model_script=None, model_object_name=None,
                 input_template=None, var_names=None, output_script=None, output_object_name=None,
                 ntasks=1, cores_per_task=1, nodes=1, resume=False, verbose=False, model_dir='Model_Runs',
                 cluster=False, fmt=None, separator=', ', **kwargs):

        # Check the platform and build appropriate call to Python
        if platform.system() in ['Windows']:
            self.python_command = "python"
        elif platform.system() in ['Darwin', 'Linux', 'Unix']:
            self.python_command = "python3"
        else:
            self.python_command = "python3"

        # Verbose option
        self.verbose = verbose

        # Format option
        self.separator = separator
        self.fmt = fmt
        if self.fmt is None:
            pass
        elif isinstance(self.fmt, str):
            if (self.fmt[0] != "{") or (self.fmt[-1] != "}") or (":" not in self.fmt):
                raise ValueError('fmt should be a string in brackets indicating a standard Python format.')
        else:
            raise TypeError('fmt should be a str.')

        # kwargs options, used only for python runs
        self.python_kwargs = kwargs

        # Input related
        self.input_template = input_template
        self.var_names = var_names
        self.n_vars = 0

        # Check if var_names is a list of strings
        if self.var_names is not None:
            if not self._is_list_of_strings(self.var_names):
                raise ValueError("Variable names should be passed as a list of strings.")

        # Model related
        self.model_dir = model_dir
        current_dir = os.getcwd()
        self.return_dir = current_dir

        # Create a list of all of the files and directories in the working directory. Do not include any other
        # directories containing the same name as model_dir
        model_files = []
        for f_name in os.listdir(current_dir):
            path = os.path.join(current_dir, f_name)
            if model_dir not in path:
                model_files.append(path)
        self.model_files = model_files

        if self.model_dir is not None:
            # Create a new directory where the model will be executed
            ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")
            work_dir = os.path.join(os.getcwd(), self.model_dir + "_" + ts)
            os.makedirs(work_dir)
            self.return_dir = work_dir

            # Copy files from the model list to model run directory
            for file_name in model_files:
                full_file_name = os.path.join(current_dir, file_name)
                if not os.path.isdir(full_file_name):
                    shutil.copy(full_file_name, work_dir)
                else:
                    new_dir_name = os.path.join(work_dir, os.path.basename(full_file_name))
                    shutil.copytree(full_file_name, new_dir_name)

            # Change current working directory to model run directory
            os.chdir(os.path.join(current_dir, work_dir))

        # Check if the model script is a python script
        model_extension = pathlib.Path(model_script).suffix
        if model_extension == '.py':
            self.model_script = model_script
        else:
            raise ValueError("The model script must be the name of a python script, with extension '.py'.")
        # Save the model object name
        self.model_object_name = model_object_name
        # Save option for resuming parallel execution
        self.resume = resume

        # Output related
        self.output_script = output_script
        self.output_object_name = output_object_name
        # Initialize a list of nsim None values. The ith position in the list will hold the qoi of the ith simulation.

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
        self.qoi_list = []
        self.nexist = 0
        self.nsim = 0

        # Check if samples are provided.
        if samples is None:
            if self.verbose:
                print("No samples. Creating the object alone.")
        elif isinstance(samples, (list, np.ndarray)):
            self.run(samples)
        else:
            raise ValueError("Samples must be passed as a list or numpy ndarray")

        # Return to current directory
        if self.model_dir is not None:
            os.chdir(current_dir)

    def run(self, samples=None, append_samples=True):
        """
        Execute a computational model with given inputs.

        This function selects the appropriate workflow depending on the type of the model (Python vs third
        party software) and the runs the model based on the type of execution (series vs parallel).

        **Input:**

        :param samples: Samples to be passed as inputs to the model. If an ndarray is passed, each row of the ndarray
                        contains one set of samples required for one execution of the model. (The first dimension of the
                        ndarray is considered to be the number of rows.) If a list is passed, each item of the list
                        contains one set of samples required for one execution of the model.

                        Default: None
        :type samples: ndarray or list

        :param append_samples: If append_samples is False, all previous samples and their responses are deleted. If
                               append samples is True, samples are appended to the existing ones.

                               Default: True
        :type append_samples: Boolean
        """

        # Ensure the input samples have the correct structure
        # --> If a list is provided, convert to at least 2d ndarray. dim1 = nsim, dim2 = n_vars
        # --> If 1D array/list is provided, convert it to a 2d array. dim1 = 1, dim2 = n_vars
        # --> If samples cannot be converted to an array, this will fail.
        samples = np.atleast_2d(samples)

        # No of simulations to be performed
        self.nsim = len(samples)

        # Number of variables
        self.n_vars = len(samples[0])

        # If append_samples is False, a new set of samples is created, the previous ones are deleted!
        if not append_samples:
            self.samples = []
            self.qoi_list = []

        # Check if samples already exist, if yes append new samples to old ones
        if not self.samples:  # There are currently no samples

            # If there are no samples, check to ensure that len(var_names) = n_vars
            if self.input_template is not None:
                if self.var_names is not None:
                    # Check to see if self.var_names has the correct length
                    if len(self.var_names) != self.n_vars:
                        raise ValueError("var_names must have the same length as the number of variables (i.e. "
                                         "len(var_names) = len(samples[0]).")
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
            self.samples = np.vstack(self.samples, samples)

        # Check if there is a template input file or not and execute the appropriate function
        if self.input_template is not None:  # If there is a template input file
            # Check if it is a file and is readable
            assert os.path.isfile(self.input_template) and os.access(self.input_template, os.R_OK), \
                "File {} doesn't exist or isn't readable".format(self.input_template)
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

        return None

    ####################################################################################################################

    def _serial_execution(self):
        """
        Perform serial execution of the model when there is a template input file

        This function loops over the number of simulations, executing the model once per loop. In each loop, the
        function creates a directory for each model run, copies files to the model run directory,
        changes the current working directory to the model run directory, calls the input function, executes the model,
        calls the output function, removes the copied files and folders, and returns to the previous directory.
        """
        if self.verbose:
            print('\nPerforming serial execution of the model with template input.\n')

        # Loop over the number of simulations, executing the model once per loop
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")
        for i in range(self.nexist, self.nexist + self.nsim):
            # Create a directory for each model run
            work_dir = os.path.join(os.getcwd(), "run_" + str(i) + '_' + ts)
            current_dir = os.getcwd()
            self._copy_files(work_dir=work_dir)

            # Change current working directory to model run directory
            os.chdir(os.path.join(current_dir, work_dir))

            # Call the input function
            self._input_serial(i)

            # Execute the model
            self._execute_serial(i)

            # Call the output function
            if self.output_script is not None:
                self._output_serial(i)

            # Remove the copied files and folders
            for file_name in self.model_files:
                full_file_name = os.path.join(work_dir, os.path.basename(file_name))
                if not os.path.isdir(full_file_name):
                    os.remove(full_file_name)
                else:
                    shutil.rmtree(full_file_name)

            # Return to the previous directory
            os.chdir(self.return_dir)

    ####################################################################################################################

    def _parallel_execution(self):
        """
        Execute the model in parallel when there is a template input file

        This function calls the input function and generates input files for all the samples, then creates a directory
        for each model run, copies files to model run directory, executes the model in parallel, collects output,
        removes the copied files and folders.
        """
        if self.verbose:
            print('\nPerforming parallel execution of the model with template input.\n')
            # Call the input function
            print('\nCreating inputs in parallel execution of the model with template input.\n')

        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")

        for i in range(self.nexist, self.nexist + self.nsim):
            # Create a directory for each model run
            work_dir = os.path.join(os.getcwd(), "run_" + str(i) + '_' + ts)
            self._copy_files(work_dir=work_dir)

        self._input_parallel(ts)

        # Execute the model
        if self.verbose:
            print('\nExecuting the model in parallel with template input.\n')

        self._execute_parallel(ts)

        # Call the output function
        if self.verbose:
            print('\nCollecting outputs in parallel execution of the model with template input.\n')

        current_dir = os.getcwd()
        for i in range(self.nsim):
            # Change current working directory to model run directory
            work_dir = os.path.join(os.getcwd(), "run_" + str(i) + '_' + ts)
            os.chdir(os.path.join(current_dir, work_dir))

            # Run output processing function
            self._output_parallel(i)

            # Remove the copied files and folders
            for file_name in self.model_files:
                full_file_name = os.path.join(work_dir, os.path.basename(file_name))
                if not os.path.isdir(full_file_name):
                    os.remove(full_file_name)
                else:
                    shutil.rmtree(full_file_name)

            # Change back to the upper directory
            os.chdir(os.path.join(work_dir, current_dir))

    ####################################################################################################################
    def _serial_python_execution(self):
        """
        Execute the python model in serial when there is no template input file

        This function imports the model object from the model script, and executes the model in series by passing the
        corresponding sample along with keyword arguments, if any, as inputs to the model object.
        """
        if self.verbose:
            print('\nPerforming serial execution of the model without template input.\n')

        # Run python model
        for i in range(self.nexist, self.nexist + self.nsim):
            sample_to_send = self.samples[i]
            model_object = getattr(self.python_model, self.model_object_name)
            if len(self.python_kwargs) == 0:
                self.model_output = model_object(sample_to_send)
            else:
                self.model_output = model_object(sample_to_send, **self.python_kwargs)
            if self.model_is_class:
                self.qoi_list[i] = self.model_output.qoi
            else:
                self.qoi_list[i] = self.model_output

    ####################################################################################################################
    def _parallel_python_execution(self):
        """
        Execute the python model in parallel when there is no template input file

        This function imports the model object from the model script, and executes the model in parallel by passing the
        samples along with keyword arguments, if any, as inputs to the model object.
        """

        if self.verbose:
            print('\nPerforming parallel execution of the model without template input.\n')
        import multiprocessing
        import UQpy.Utilities as Utilities

        sample = []
        pool = multiprocessing.Pool(processes=self.ntasks)
        for i in range(self.nsim):
            sample_to_send = np.atleast_2d(self.samples[i + self.nexist])
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
        exec('from ' + self.output_script[:-3] + ' import ' + self.output_object_name)
        self.model_output = eval(self.output_object_name + '(index)')
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
            print('Created ' + str(self.nsim) + ' input files in the directory ./InputFiles. \n')

    def _execute_parallel(self, timestamp):
        """
        Build the command string and execute the model in parallel using subprocess and gnu parallel

        ** Input: **

        :param timestamp: Timestamp which is appended to the name of the input file
        :type timestamp: str
        """
        # TODO: LOHIT - Generalize run_string and parallel_string for any cluster
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

        # If running on MARCC cluster
        if self.cluster:
            self.srun_string = "srun -N " + str(self.nodes) + " -n1 -c" + str(self.cores_per_task) + " --exclusive "
            # self.srun_string = "sbatch -N" + str(self.nodes) + " -n1 -c" + str(self.cores_per_task) + " --exclusive "
            self.model_command_string = (
                    self.parallel_string + self.srun_string + "'cd run_{1}_" + timestamp + " && " + self.python_command
                    + " -u " + str(self.model_script) + "' {1}  ::: {0.." + str(self.nsim - 1) + "}")
        else:  # If running locally
            self.model_command_string = (self.parallel_string + " 'cd run_{1}_" + timestamp + "&& " +
                                         self.python_command + " -u " +
                                         str(self.model_script) + "' {1}  ::: {0.." + str(self.nsim - 1) + "}")

        print(self.model_command_string)
        subprocess.run(self.model_command_string, shell=True)

    def _output_parallel(self, index):
        """
        Extract output from parallel execution

        ** Input: **

        :param index: The simulation number
        :type index: int
        """

        self._output_serial(index)

    ####################################################################################################################
    # Helper functions
    @staticmethod
    def _create_input_files(file_name, num, text, new_folder='InputFiles'):
        """
        Create input files using filename, index, text

        ** Input: **

        :param file_name: Name of index file
        :type file_name: str

        :param num: The simulation number
        :type num: int

        :param text: Content of input file
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

        :param var_names: Name of the probabilistic input variables.
        :type var_names: list of str

        :param samples: Samples values of the input variables.
        :type samples: ndarray

        :param template_text: Text in the template input file, with placeholders where the sample values of the input
                              variables need to be placed.
        :type template_text: str

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
                        print("Index Error: {0}\n".format(err))
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
                            "Found " + str(count) + " instances of variable: '" + var_names[j] + "' in the input file.")
                    else:
                        print(
                            "Found " + str(count) + " instance of variable: '" + var_names[j] + "' in the input file.")
        return new_text

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
                "A python model should be defined as a function or class in the script.")

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
                    print('The model class that will be run: ' + self.model_object_name)
                self.model_is_class = True
            elif self.model_object_name in function_list:
                if self.verbose:
                    print('The model function that will be run: ' + self.model_object_name)
                self.model_is_class = False
            else:
                if self.model_object_name is None:
                    raise ValueError("There are more than one objects in the module. Specify the name of the function "
                                     "or class which has to be executed.")
                else:
                    print('You specified the model object name as: ' + str(self.model_object_name))
                    raise ValueError("The file does not contain an object which was specified as the model.")

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
                "A python model should be defined as a function or class in the script.")

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
                    print('The output class that will be run: ' + self.output_object_name)
                self.output_is_class = True
            elif self.output_object_name in function_list:
                if self.verbose:
                    print('The output function that will be run: ' + self.output_object_name)
                self.output_is_class = False
            else:
                if self.output_object_name is None:
                    raise ValueError("There are more than one objects in the module. Specify the name of the function "
                                     "or class which has to be executed.")
                else:
                    print('You specified the output object name as: ' + str(self.output_object_name))
                    raise ValueError("The file does not contain an object which was specified as the output processor.")

    def _copy_files(self, work_dir):
        os.makedirs(work_dir)
        # Copy files from the model list to model run directory
        current_dir = os.getcwd()
        for file_name in self.model_files:
            full_file_name = os.path.join(current_dir, file_name)
            if not os.path.isdir(full_file_name):
                shutil.copy(full_file_name, work_dir)
            else:
                new_dir_name = os.path.join(work_dir, os.path.basename(full_file_name))
                shutil.copytree(full_file_name, new_dir_name)
