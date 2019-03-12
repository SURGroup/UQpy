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


import os
import subprocess
import pathlib
import re
import collections
import numpy as np
import datetime
import shutil


class RunModel:
    """
    Run a computational model at specified sample points.

    This class is the interface between UQpy and computational models. The model is called in a Python script whose
    name must be passed as one the arguments to the RunModel call. If the model is in Python, UQpy can interface with
    the model without the need for an input file. In this case, UQpy imports the model module and executes the model
    object. If the model is not in Python, RunModel must be provided the name of a template input file and an output
    Python script along with the name of the Python script that runs the model.


    :param samples: Samples to be passed as inputs to the model. Samples can be passed either as an ndarray or a list.
    If an ndarray is passed, each row of the ndarray contains one set of samples required for one execution of the
    model. (The first dimension of the ndarray is considered to be the number of rows.)
    If a list is passed, each item of the list contains one set of samples required for one execution of the model.
    :type samples: ndarray or list

    :param model_script: The filename (with extension) of the Python script which contains commands to execute the
    model. The model script must be present in the current working directory from which RunModel is called.
    :type model_script: str

    :param model_object_name: In the Python model workflow, model_object_name specifies the name of the function or
    class within model_script which executes the model. If there is only one function or class in the model_script, then
    it is not necessary to specify the model_object_name. If there are multiple objects within the model_script, then
    model_object_name must be specified.
    model_object_name is not used in the third-party software model workflow.
    :type model_object_name: str

    :param input_template: The name of the template input file which will be used to generate input files for each
    run of the model. When operating RunModel with a third-party software model, input_template must be specified.
    input_template is not used in the Python model workflow.
    :type input_template: str

    :param var_names: A list containing the names of the variables present in the template input file. If an
    input template is provided and a list of variable names is not passed, ie if var_names=None, then the default
    variable names x0, x1, x2,...,xn are created and used by RunModel, where n is the number of variables. The
    number of variables is equal to the shape of the first row if samples is passed as an ndarray or the shape of the
    first item if samples is passed as a list.
    varnamesis not used in the Python model workflow.
    :type var_names: list of str or None

    :param output_script: The filename of the Python script which contains the commands to process the output from
    third-party software model evaluation. The output_script is used to return the output quantities of interest to
    RunModel for subsequent UQpy processing (e.g. for adaptive methods that utilize the results of previous simualtions
    to initialize new simulations).
    output_script is not used in the Python model workflow. In the Python model workflow, all model postprocessing is
    handled within model_script.
    If, in the third-party software model workflow, output_script = None (the default), then RunModel.qoi_list is empty
    and postprocessing must be handled outside of UQpy.
    :type output_script: str

    :param output_object_name: The name of the function or class that is used to collect the output values from
    third-party software model output files.
    If the object is aclass named cls, for example, the output must be saved as cls.qoi. If it is a function, it should
    return the output quantity of interest. If there is only one function or only one class in output_script, then it is
    not necessary to specify output_object_name. If there are multiple objects in output_script, then output_object_name
    must be specified.
    outputobjectname is not used in the Python model workflow.
    :type output_object_name: str

    :param ntasks: Number of tasks to be run in parallel.
    By default, ntasks = 1 and the models are executed serially. Setting ntasks equal to a positive integer greater than
    1 will trigger the parallel workflow.
    RunModel uses GNU parallel to execute models which require an input template in parallel and the concurrent module
    to execute Python models in parallel.
    :type ntasks: int

    :param cores_per_task: Number of cores to be used by each task.
    In cases where a third-party model runs across multiple CPUs, this optional attribute allocates the necessary
    resources to each model evaluation.
    cores_per_task is not used in the Python model workflow
    :type cores_per_task: int

    :param nodes: Number of nodes across which to distribute parallel jobs on an HPC cluster in the third-party software
    model workflow.
    If more than one compute node is required to execute the runs in parallel, nodes must be specified. For example, on
    the Maryland Advanced Research Computing Center (MARCC), an HPC shared by Johns Hopkins University and the
    University of Maryland, each compute node has 24 cores. To run an analysis with more than 24 parallel jobs on MARCC
    requires nodes > 1.
    nodes is not used in the Python model workflow.

    :type nodes: int

    :param resume: If resume = True, GNU parallel enables UQpy to resume execution of any model evaluations that failed
    to execute in the third-party software model workflow.
    To use this feature, execute the same call to RunModel which failed to complete but with resume = True.  The same
    set of samples must be passed to resume processing from the last successful execution of the model.
    resume is not used in the Python model workflow.
    :type resume: Boolean

    :param verbose: Set verbose = True if you want RunModel to print status messages to the terminal during execution.
    verbose = False by default.
    :type verbose: Boolean

    :param model_dir: Specifies the name of the sub-directory from which the model will be executed and to which output
    files will be saved.
    model_dir = None by default, which results in model execution from the Python current working directory. If
    model_dir is passed a string, then a new directory is created by RunModel within the current directory whose name is
    model_dir appended with a timestamp.
    :type model_dir: str

    :param cluster: Set cluster = True if executing on an HPC cluster. Setting cluster = True enables RunModel to
    execute the model using the necessary SLURM commands. cluster = False by default.
    RunModel is configured for HPC clusters that operate with the SLURM scheduler. In order to execute a third-party
    model with RunModel on an HPC cluster, the HPC must use SLURM.
    cluster is not used for the Python model workflow.
    :type cluster: Boolean

    Output:
    :return: RunModel.qoi_list: A list containing the output quantities of interest extracted from the model output
    files by output_script. This is a list of length equal to the number of simulations. Each item of this list contains
    the quantity of interest from the associated simulation.
    :rtype: RunModel.qoi_list: list
    """

    def __init__(self, samples=None, model_script=None, model_object_name=None,
                 input_template=None, var_names=None, output_script=None, output_object_name=None,
                 ntasks=1, cores_per_task=1, nodes=1, resume=False, verbose=False, model_dir=None,
                 cluster=False):

        # Check if samples are provided
        if samples is None:
            raise ValueError('Samples must be provided as input to RunModel.')
        elif isinstance(samples, (list, np.ndarray)):
            self.samples = samples
            self.nsim = len(self.samples)  # This assumes that the number of rows is the number of simulations.
        else:
            raise ValueError("Samples must be passed as a list or numpy ndarray")

        # Verbose option
        self.verbose = verbose

        # Input related
        self.input_template = input_template
        self.var_names = var_names
        # Check if var_names is a list of strings
        if self.var_names is not None:
            if self._is_list_of_strings(self.var_names):
                self.n_vars = len(self.var_names)
            else:
                raise ValueError("Variable names should be passed as a list of strings.")
        elif self.input_template is not None:
            # If var_names is not passed and there is an input template, create default variable names
            nvars = samples[0].shape[0]
            self.var_names = []
            for i in range(nvars):
                self.var_names.append('x%d' % i)

        # Model related
        self.model_dir = model_dir
        current_dir = os.getcwd()
        self.return_dir = current_dir

        # Create a list of all of the files and directories in the working directory
        model_files = list()
        for f_name in os.listdir(current_dir):
            path = os.path.join(current_dir, f_name)
            # if not os.path.isdir(path):
            #     model_files.append(path)
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
        self.qoi_list = [None] * self.nsim

        # Number of tasks
        self.ntasks = ntasks
        # Number of cores_per_task
        self.cores_per_task = cores_per_task
        # Number of nodes
        self.nodes = nodes

        # If running on cluster or not
        self.cluster = cluster

        # Check if there is a template input file or not and execute the appropriate function
        if self.input_template is not None:  # If there is a template input file
            # Check if it is a file and is readable
            assert os.path.isfile(self.input_template) and os.access(self.input_template, os.R_OK), \
                "File {} doesn't exist or isn't readable".format(self.input_template)
            # Read in the text from the template file
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

        # Return to current directory
        if self.model_dir is not None:
            os.chdir(current_dir)

    ####################################################################################################################
    def _serial_execution(self):
        """
        Perform serial execution of the model when there is a template input file
        :return:
        """
        if self.verbose:
            print('\nPerforming serial execution of the model with template input.\n')

        # Loop over the number of simulations, executing the model once per loop
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")
        for i in range(self.nsim):
            # Create a directory for each model run
            work_dir = os.path.join(os.getcwd(), "run_" + str(i) + '_' + ts)
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
        :return:
        """
        if self.verbose:
            print('\nPerforming parallel execution of the model with template input.\n')
            # Call the input function
            print('\nCreating inputs in parallel execution of the model with template input.\n')

        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f_%p")

        for i in range(self.nsim):
            # Create a directory for each model run
            work_dir = os.path.join(os.getcwd(), "run_" + str(i) + '_' + ts)
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
        :return:
        """
        if self.verbose:
            print('\nPerforming serial execution of the model without template input.\n')

        # Run python model
        for i in range(self.nsim):
            exec('from ' + self.model_script[:-3] + ' import ' + self.model_object_name)
            if isinstance(self.samples, list):
                sample_to_send = self.samples[i]
            elif isinstance(self.samples, np.ndarray):
                sample_to_send = self.samples[None, i]
            # self.model_output = eval(self.model_object_name + '(self.samples[i])')
            self.model_output = eval(self.model_object_name + '(sample_to_send)')
            if self.model_is_class:
                self.qoi_list[i] = self.model_output.qoi
            else:
                self.qoi_list[i] = self.model_output

    ####################################################################################################################
    def _parallel_python_execution(self):
        """
        Execute the python model in parallel when there is no template input file
        :return:
        """
        if self.verbose:
            print('\nPerforming parallel execution of the model without template input.\n')
        # self._serial_python_execution()
        import concurrent.futures
        # Try processes # Does not work - raises TypeError: can't pickle module objects
        # indices = range(self.nsim)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     for index, res in zip(indices, executor.map(self._run_parallel_python, self.samples)):
        #         self.qoi_list[index] = res

        # Try threads - this works but is slow
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.ntasks) as executor:
            index = 0
            for sample in self.samples:
                res = {executor.submit(self._run_parallel_python, sample): index}
                for future in concurrent.futures.as_completed(res):
                    resnum = res[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (resnum, exc))
                    else:
                        self.qoi_list[index] = data
                index += 1

        # from multiprocessing import Process
        # from multiprocessing import Queue
        #
        # # Initialize the parallel processing queue and processes
        # que = Queue()
        # jobs = [Process(target=self._run_parallel_python_chunked,
        #                 args=([self.samples[index*self.ntasks:(index+1)*self.ntasks-1]]))
        #         for index in range(self.ntasks)]
        # # Start the parallel processes.
        # for j in jobs:
        #     j.start()
        # for j in jobs:
        #     j.join()
        #
        # # Collect the results from the processes and sort them into the original sample order.
        # results = [que.get(j) for j in jobs]
        # for i in range(self.nsim):
        #     k = 0
        #     for j in results[i][0]:
        #         self.qoi_list[j] = results[i][1][k]
        #         k = k + 1

    def _run_parallel_python(self, sample):
        """
        Execute the python model in parallel
        :param sample: One sample point where the model has to be evaluated
        :return:
        """
        exec('from ' + self.model_script[:-3] + ' import ' + self.model_object_name)
        parallel_output = eval(self.model_object_name + '(sample)')
        if self.model_is_class:
            par_res = parallel_output.qoi
        else:
            par_res = parallel_output

        return par_res

    ####################################################################################################################
    def _input_serial(self, index):
        """
        Create one input file using the template and attach the index to the filename
        :param index: The simulation number
        :return:
        """
        # Create new text to write to file
        self.new_text = self._find_and_replace_var_names_with_values(var_names=self.var_names,
                                                                     samples=self.samples[index],
                                                                     template_text=self.template_text,
                                                                     index=index,
                                                                     user_format='{:.4E}')
        # Write the new text to the input file
        self._create_input_files(file_name=self.input_template, num=index, text=self.new_text,
                                 new_folder='InputFiles')

    def _execute_serial(self, index):
        """
        Execute the model once using the input file of index number
        :param index: The simulation number
        :return:
        """
        self.model_command = (["python3", str(self.model_script), str(index)])
        subprocess.run(self.model_command)

    def _output_serial(self, index):
        """
        Execute the output script, obtain the output qoi and save it in qoi_list
        :param index: The simulation number
        :return:
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
        :return:
        """
        # Loop over the number of samples and create input files in a folder in current directory
        for i in range(self.nsim):
            # Create new text to write to file
            new_text = self._find_and_replace_var_names_with_values(var_names=self.var_names,
                                                                    samples=self.samples[i],
                                                                    template_text=self.template_text,
                                                                    index=i,
                                                                    user_format='{:.4E}')
            folder_to_write = 'run_' + str(i) + '_' + timestamp + '/InputFiles'
            # Write the new text to the input file
            self._create_input_files(file_name=self.input_template, num=i, text=new_text,
                                     new_folder=folder_to_write)
        if self.verbose:
            print('Created ' + str(self.nsim) + ' input files in the directory ./InputFiles. \n')

    def _execute_parallel(self, timestamp):
        """
        Build the command string and execute the model in parallel using subprocess and gnu parallel
        :return:
        """
        # Check if logs folder exists, if not, create it
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # If the user sets resume=True, do not delete log file. Else, delete logfile before running
        if self.resume is False:
            try:
                os.remove("logs/runtask.log")
            except OSError:
                pass
        self.parallel_string = "parallel --delay 0.2 --joblog logs/runtask.log --resume -j " + str(self.ntasks)

        # If running on MARCC cluster
        if self.cluster:
            self.srun_string = "srun -N " + str(self.nodes) + " -n1 -c" + str(self.cores_per_task) + " --exclusive"
            self.model_command_string = (
                    self.parallel_string + self.srun_string + " 'cd run_{1}_" + timestamp + "&& python3 -u " +
                    str(self.model_script) + "' {1}  ::: {0.." + str(self.nsim - 1) + "}")
            # self.model_command_string = (self.parallel_string + self.srun_string + " 'python3 -u " +
            #                              str(self.model_script) + "' {1} ::: {0.." + str(self.nsim - 1) + "}")
        else:  # If running locally
            self.model_command_string = (self.parallel_string + " 'cd run_{1}_" + timestamp + "&& python3 -u " +
                                         str(self.model_script) + "' {1}  ::: {0.." + str(self.nsim - 1) + "}")
            # self.model_command_string = (self.parallel_string + " '" + "cd " + destination + "| pwd | python3 -u " +
            #                              str(self.model_script) + "' {1} ::: {0.." + str(self.nsim - 1) + "}")

        # self.model_command = shlex.split(self.model_command_string)
        # subprocess.run(self.model_command)

        subprocess.run(self.model_command_string, shell=True)

    def _output_parallel(self, index):
        """
        Extract output from parallel execution
        :param index: The simulation number
        :return:
        """
        self._output_serial(index)

    ####################################################################################################################
    # Helper functions
    def _create_input_files(self, file_name, num, text, new_folder='InputFiles'):
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        base_name = os.path.splitext(os.path.basename(file_name))
        new_name = os.path.join(new_folder, base_name[0] + "_" + str(num) + base_name[1])
        with open(new_name, 'w') as f:
            f.write(text)
        return

    def _find_and_replace_var_names_with_values(self, var_names, samples, template_text, index, user_format='{:.4E}'):
        # TODO: deal with cases which have both var1 and var11
        new_text = template_text
        for j in range(len(var_names)):
            string_regex = re.compile(r"<" + var_names[j] + r".*?>")
            count = 0
            for string in string_regex.findall(template_text):
                temp = string.replace(var_names[j], "samples[" + str(j) + "]")
                temp = eval(temp[1:-1])
                if isinstance(temp, collections.Iterable):
                    temp = np.array(temp).flatten()
                    to_add = ''
                    for i in range(len(temp) - 1):
                        to_add += str(temp[i]) + ', '
                    to_add += str(temp[-1])
                else:
                    to_add = str(temp)
                new_text = new_text[0:new_text.index(string)] + to_add \
                           + new_text[(new_text.index(string) + len(string)):]
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

    def _is_list_of_strings(self, lst):
        return bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)

    def _check_python_model(self):
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

    ####################################################################################################################
    # Unused functions
    def _collect_output(self, qoi_list, qoi_output, pos):
        qoi_list[pos] = qoi_output
        return qoi_list

    # def _run_parallel_python_chunked(self, some_samples):
    #     par_res = [[] for i in range(some_samples.shape[0])]
    #     for i in range(some_samples.shape[0]):
    #         exec('from ' + self.model_script[:-3] + ' import ' + self.model_object_name)
    #         parallel_output = eval(self.model_object_name + '(some_samples[i])')
    #         if self.model_is_class:
    #             par_res[i] = parallel_output.qoi
    #         else:
    #             par_res[i] = parallel_output
    #
    #     return par_res

# """This module contains functionality for the run model method supported in UQpy."""
#
# import os
# import numpy as np
# import sys
#
#
# class RunModel:
#
#     """
#     A class used to run a computational model a specified sample points.
#
#     This class takes samples, either passed as a variable or read through a text file, and runs a specified
#     computational model at those sample points. This can be done by either passing variables and running entirely in
#     python or by calling shell scripts that run a third-party software model.
#
#     Input:
#     :param samples: The sample values at which the model will be evaluated. Samples can be passed directly as an array
#                         or can be passed through the text file 'UQpy_Samples.txt'. If passing samples via text file, set
#                         samples = None or do not set the samples input.
#     :type samples: ndarray
#
#     :param dimension: The dimension of the random variable whose samples are being passed to the model.
#     :type dimension: int
#
#     :param model_type: Define the model as a python file or as a third party software model (e.g. Matlab, Abaqus, etc.)
#             Options: None - Run a third party software model
#                      'python' - Run a python model. When selected, the python file must contain a class RunPythonModel
#                                 that takes, as input, samples and dimension and returns quantity of interest (qoi) in
#                                 in list form where there is one item in the list per sample. Each item in the qoi list
#                                 may take type the user prefers.
#             Default: None
#     :type model_type: str
#
#     :param model_script: Defines the script (must be either a shell script (.sh) or a python script (.py)) used to call
#                             the model.
#                          This is a user-defined script that must be provided.
#                          If model_type = 'python', this must be a python script (.py) having a specified class
#                             structure. Details on this structure can be found in the UQpy documentation.
#     :type: model_script: str
#
#     :param input_script: Defines the script (must be either a shell script (.sh) or a python script (.py)) that takes
#                             samples generated by UQpy from the sample file generated by UQpy (UQpy_run_{0}.txt) and
#                             imports them into a usable input file for the third party solver. Details on
#                             UQpy_run_{0}.txt can be found in the UQpy documentation.
#                          If model_type = None, this is a user-defined script that the user must provide.
#                          If model_type = 'python', this is not used.
#     :type: input_script: str
#
#     :param output_script: (Optional) Defines the script (must be either a shell script (.sh) or python script (.py))
#                             that extracts quantities of interest from third-party output files and saves them to a file
#                             (UQpy_eval_{}.txt) that can be read for postprocessing and adaptive sampling methods by
#                             UQpy.
#                           If model_type = None, this is an optional user-defined script. If not provided, all run files
#                             and output files will be saved in the folder 'UQpyOut' placed in the current working
#                             directory. If provided, the text files UQpy_eval_{}.txt are placed in this directory and all
#                             other files are deleted.
#                           If model_type = 'python', this is not used.
#     :type output_script: str
#
#     :param cpu: Number of CPUs over which to run the job.
#                 UQpy distributes the total number of model evaluations over this number of CPUs
#                 Default: 1 - Runs serially
#     :type cpu: int
#
#     Output:
#     :return model_eval: An instance of a sub-class that contains the model solutions. Depending on how the model
#                             is run, model_eval is an instance of a different class.
#                         If model_type = 'python', model_eval is an instance of the class RunPythonModel defined in the
#                             python model_script.
#                         If model_type = 'None' and cpu <= 1, model_eval is an instance of the class RunSerial
#                         If model_type = 'None' and cpu > 1, model_eval is an instance of the class RunParallel
#                         Regardless of model_type, model_eval has the following key attributes:
#                             model_eval.samples = Sample values at which the model has been evaluated.
#                             model_eval.QOI = Solution of the model at each sample value.
#     :rtype: model_eval: list
#                         In general it is a list. The two key attributes of model_eval have the following type:
#                                 model_eval.samples = numpy array
#                                 model_eval.QOI = list
#     """
#
#     # Authors: Dimitris G. Giovanis, Michael D. Shields
#     # Last Modified: 6/7/18 by Dimitris G. Giovanis
#
#     def __init__(self, samples=None, dimension=None, model_type=None, model_script=None, input_script=None,
#                  output_script=None,  cpu=None):
#
#         self.CPUs = cpu
#         self.model_type = model_type
#         self.model_script = model_script
#         self.input_script = input_script
#         self.output_script = output_script
#         self.dimension = dimension
#         self.model_eval = list
#
#         # If samples=None, then samples must be imported from UQpy_Samples.txt. Load the file and assign the samples to
#         # 'self.samples'. Otherwise, read the samples as input.
#         if samples is None and os.path.isfile('UQpy_Samples.txt'):
#             self.samples = np.loadtxt('UQpy_Samples.txt', dtype=np.float32)
#             if self.samples.ndim == 1:
#                 if self.dimension == 1:
#                     self.samples = self.samples.reshape(self.samples.shape[0], self.dimension)
#                 else:
#                     self.samples = self.samples.reshape(1, self.samples.shape[0])
#         elif samples is not None:
#             self.samples = samples
#         else:
#             raise ValueError('Samples must be provided either as input to RunModel or from UQpy_Samples.txt')
#
#         ################################################################################################################
#         # Run a python model by importing the user-specified model_script file
#         # model_script must contain a class RunPythonModel - See documentation for details.
#         if self.model_type == 'python':
#             # Check that the script is a python file
#             if not self.model_script.lower().endswith('.py'):
#                 raise ValueError('A python script, with extension .py, must be provided.')
#
#             model_script = self.model_script[:-3]
#             python_model = __import__(model_script)
#             print("\nEvaluating the model...\n")
#             self.model_eval = python_model.RunPythonModel(self.samples, self.dimension)
#
#         ################################################################################################################
#         # Run a third-party software model with file-passing
#         elif self.model_type is None:
#             import shutil
#             current_dir = os.getcwd()
#
#             ############################################################################################################
#             # Create a unique temporary directory 'tmp'.
#             # Run the model from this directory.
#             # Move the data generated by the model to the directory 'UQpyOut'
#             # Remove 'tmp' after completion.
#             folder_name = 'UQpyOut'
#             output_directory = os.path.join(current_dir, folder_name)
#
#             # Create a list of all of the working files
#             model_files = list()
#             for f_name in os.listdir(current_dir):
#                 path = os.path.join(current_dir, f_name)
#                 if not os.path.isdir(path):
#                     model_files.append(path)
#
#             # Create tmp directory
#             dir_path = os.path.join(current_dir, 'tmp')
#             if os.path.exists(dir_path) and os.path.isdir(dir_path):
#                 shutil.rmtree(dir_path)
#             os.makedirs('tmp', exist_ok=False)
#             work_dir = os.path.join(current_dir, 'tmp')
#
#             # Copy files from the model list to tmp
#             for file_name in model_files:
#                 full_file_name = os.path.join(current_dir, file_name)
#                 shutil.copy(full_file_name, work_dir)
#
#             # Change current working directory to tmp
#             os.chdir(os.path.join(current_dir, work_dir))
#
#             # Check for parallel or serial processing
#             if self.CPUs != 1 and self.CPUs is not None:
#                 parallel_processing = True
#                 import multiprocessing
#                 n_cpu = multiprocessing.cpu_count()
#                 if self.CPUs > n_cpu:
#                     print("Error: You have available {0:1d} CPUs. Start parallel computing ..."
#                           "using {0:1d} CPUs".format(n_cpu))
#                     self.CPUs = n_cpu
#             else:
#                 parallel_processing = False
#
#             # Run the model
#             print("\nEvaluating the model...\n")
#             if parallel_processing is True:
#                 self.model_eval = self.RunParallel(samples=self.samples, cpu=self.CPUs, dimension=self.dimension,
#                                                    model_script=self.model_script, input_script=self.input_script,
#                                                    output_script=self.output_script)
#             else:
#                 if self.output_script is not None:
#                     self.model_eval = self.RunSerial(samples=self.samples, dimension=self.dimension,
#                                                      model_script=self.model_script, input_script=self.input_script,
#                                                      output_script=self.output_script)
#                 else:
#                     self.RunSerial(self.samples)
#
#             # Move the data to directory UQpyOut
#             os.makedirs(output_directory, exist_ok=True)
#
#             path = os.path.join(current_dir, work_dir)
#
#             if self.output_script is not None:
#                 src_files = [filename for filename in os.listdir(path) if filename.startswith("UQpy_eval_")]
#
#                 for file_name in src_files:
#                     full_file_name = os.path.join(path, file_name)
#                     shutil.copy(full_file_name, output_directory)
#             else:
#                 src_files = [filename for filename in os.listdir(path)]
#
#                 for file_name in src_files:
#                     full_file_name = os.path.join(path, file_name)
#                     shutil.copy(full_file_name, output_directory)
#
#             # Delete the tmp working directory
#             shutil.rmtree(work_dir)
#             os.chdir(current_dir)
#
#     ####################################################################################################################
#     class RunSerial:
#
#         """
#         A subclass of RunModel to run a third-party software model serially (without parallel processing).
#
#         Most attributes of this subclass are inherited from RunModel. The only variable that is not inherited is QOI.
#
#         Input:
#         :param samples: Inherited from RunModel. See its documentation.
#         :type samples: ndarray
#
#         :param dimension: Inherited from RunModel. See its documentation.
#         :type dimension: int
#
#         :param model_script: Inherited from RunModel. See its documentation.
#         :type: model_script: str
#
#         :param input_script: Inherited from RunModel. See its documentation.
#         :type: input_script: str
#
#         :param output_script: Inherited from RunModel. See its documentation.
#         :type output_script: str
#
#         Output:
#         :return QOI: List containing the Quantity of Interest from the simulations
#                     Each item in the list corresponds to one simulation
#         :rtype QOI: list
#                     Each item in the list may be of arbitrary data type (e.g. int, float, ndarray, etc.)
#         """
#
#         # Authors: Dimitris G. Giovanis, Michael D. Shields
#         # Updated: 6/7/18 by Michael D. Shields & Dimitris G. Giovanis
#
#         def __init__(self, samples=None, dimension=None, model_script=None, input_script=None, output_script=None):
#
#             self.dimension = dimension
#             self.model_script = model_script
#             self.input_script = input_script
#             self.output_script = output_script
#             self.samples = samples
#             self.QOI = list()
#
#             for i in range(self.samples.shape[0]):
#
#                 # Write each value of UQpy_Samples.txt (self.samples) into a separate *.txt file
#                 with open('UQpy_run_{0}.txt'.format(i), 'wb') as f:
#                     np.savetxt(f, self.samples[i, :], fmt='%0.5f')
#
#                 # Run input_script to create the input file for the model
#                 # input_script is a user defined script that converts UQpy_run_{0}.txt into a usable input file for
#                 #   third party software.
#                 if self.input_script.lower().endswith('.sh'):
#                     join_input_script = './{0} {1}'.format(self.input_script, i)
#                     os.system(join_input_script)
#                 elif self.input_script.lower().endswith('.py'):
#                     join_input_script = 'python {0} {1}'.format(self.input_script, i)
#                     os.system(join_input_script)
#                 else:
#                     print('Unrecognized script type. Options are shell script (.sh) or python script (.py).')
#                     sys.exit()
#
#                 # Run model_script to run the model
#                 # model_script is a user defined script that calls the third-party model.
#                 if self.model_script.lower().endswith('.sh'):
#                     join_model_script = './{0} {1}'.format(self.model_script, i)
#                     os.system(join_model_script)
#                 elif self.model_script.lower().endswith('.py'):
#                     join_model_script = 'python {0} {1}'.format(self.model_script, i)
#                     os.system(join_model_script)
#                 else:
#                     print('Unrecognized script type. Options are shell script (.sh) or python script (.py).')
#                     sys.exit()
#
#                 if self.output_script is not None:
#                     # Run output_script to extract output of interest from model output files.
#                     if self.output_script.lower().endswith('.sh'):
#                         join_output_script = './{0} {1}'.format(self.output_script, i)
#                         os.system(join_output_script)
#                     elif self.output_script.lower().endswith('.py'):
#                         join_output_script = 'python {0} {1}'.format(self.output_script, i)
#                         os.system(join_output_script)
#                     else:
#                         print('Unrecognized script type. Options are shell script (.sh) or python script (.py).')
#                         sys.exit()
#
#                     # Save the results from each simulation to a separate text file
#                     self.QOI.append(np.loadtxt('UQpy_eval_{}.txt'.format(i)))
#
#     ####################################################################################################################
#     class RunParallel:
#
#         """
#         A subclass of RunModel to run a third-party software model with parallel processing.
#
#         Most attributes of this subclass are inhereted from RunModel. The only variable that is not inherited is QOI.
#
#         Input:
#         :param samples: Inherited from RunModel. See its documentation.
#         :type samples: ndarray
#
#         :param dimension: Inherited from RunModel. See its documentation.
#         :type dimension: int
#
#         :param model_script: Inherited from RunModel. See its documentation.
#         :type: model_script: str
#
#         :param input_script: Inherited from RunModel. See its documentation.
#         :type: input_script: str
#
#         :param output_script: Inherited from RunModel. See its documentation.
#         :type output_script: str
#
#         Output:
#         :return QOI: List containing the Quantity of Interest from the simulations
#                     Each item in the list corresponds to one simulation
#         :rtype QOI: list
#                     Each item in the list may be of arbitrary data type (e.g. int, float, ndarray, etc.)
#         """
#
#         # Authors: Dimitris Giovanis, Michael D. Shields
#         # Updated: 5/1/18 by Michael D. Shields & Dimitris Giovanis
#
#         def __init__(self, samples=None, cpu=None, model_script=None, input_script=None, output_script=None,
#                      dimension=None):
#
#             self.samples = samples
#             self.dimension = dimension
#             self.CPUs = cpu
#             self.model_script = model_script
#             self.input_script = input_script
#             self.output_script = output_script
#
#             from multiprocessing import Process
#             from multiprocessing import Queue
#
#             jobs_per_cpu = int(np.floor(self.samples.shape[0]/self.CPUs))
#             jobs_remaining = np.mod(self.samples.shape[0], self.CPUs)
#             if jobs_per_cpu == 0:
#                 self.CPUs = jobs_remaining
#                 print('The number of CPUs used is {}\n '.format(self.CPUs))
#
#             # Break the simulation set into sets to be run over each CPU
#             batch_ind = self.chunk_samples_cpu()
#
#             # Initialize the parallel processing queue and processes
#             que = Queue()
#             jobs = [Process(target=self.run_parallel_model, args=(batch_ind[a], que)) for a in
#                     range(self.CPUs)]
#
#             # Start the parallel processes.
#             for j in jobs:
#                 j.start()
#             for j in jobs:
#                 j.join()
#
#             # Collect the results from the processes and sort them into the original sample order.
#             self.QOI = [None]*self.samples.shape[0]
#             results = [que.get(j) for j in jobs]
#             for i in range(self.CPUs):
#                 k = 0
#                 for j in results[i][0]:
#                     self.QOI[j] = results[i][1][k]
#                     k = k+1
#
#         ################################################################################################################
#         # Function to call the model
#         def run_parallel_model(self, job_index, que):
#
#             if self.samples.size == 1:
#                 self.samples = self.samples.reshape(1, 1)
#
#             if len(self.samples.shape) == 1 and self.dimension != 1:
#                 self.samples = self.samples.reshape(1, self.samples.shape[0])
#             elif len(self.samples.shape) == 1 and self.dimension == 1:
#                 self.samples = self.samples.reshape(self.samples.shape[0], 1)
#
#             model_eval = list()
#             for i in job_index:
#
#                 # Write each value of UQpyOut.txt into a *.txt file
#                 np.savetxt('UQpy_run_{0}.txt'.format(int(i)), self.samples[i, :], newline=' ', delimiter=',',
#                            fmt='%0.5f')
#
#                 # Run the input script to convert UQpy_run_X.txt to a valid model input
#                 if self.input_script.lower().endswith('.sh'):
#                     join_input_script = './{0} {1}'.format(self.input_script, int(i))
#                     os.system(join_input_script)
#                 else:
#                     print('Unrecognized type of input file script. Must be .sh')
#                     sys.exit()
#
#                 # Run the third-party model script (model_script)
#                 if self.model_script.lower().endswith('.sh'):
#                     join_model_script = './{0} {1}'.format(self.model_script, int(i))
#                     os.system(join_model_script)
#                 else:
#                     print('Unrecognized type of model file')
#                     sys.exit()
#
#                 # Run the output script to convert model results to UQpy_eval_X.txt to be read by UQpy
#                 if self.output_script.lower().endswith('.sh'):
#                     join_output_script = './{0} {1}'.format(self.output_script, int(i))
#                     os.system(join_output_script)
#                 else:
#                     print('Unrecognized type of Input file')
#                     sys.exit()
#
#                 model_eval.append(np.loadtxt('UQpy_eval_{0}.txt'.format(int(i))))
#
#                 src_files = 'UQpy_eval_{0}.txt'.format(int(i))
#                 file_new = src_files.replace("UQpy_eval_{0}.txt".format(int(i)), "Model_{0}.txt".format(int(i)))
#                 os.rename(src_files, file_new)
#
#             model_eval = que.put([job_index, model_eval])
#
#             return model_eval
#
#         ################################################################################################################
#         # Chunk the samples into batches
#
#         def chunk_samples_cpu(self):
#
#             size_ = np.array([np.ceil(self.samples.shape[0]/self.CPUs)]).astype(int)
#             dif = np.sum(size_) - self.samples.shape[0]
#             for k in range(dif):
#                 size_[k] = size_[k] - 1
#             batches = np.zeros(self.CPUs)
#             batch_ind = np.zeros(self.CPUs)
#             for i in range(self.CPUs):
#                 if i == 0:
#                     batch_ind[i] = range(size_[i])
#                 else:
#                     batch_ind[i] = range(int(np.sum(size_[:i])), int(np.sum(size_[:i+1])))
#                 batches[i] = self.samples[batch_ind[i], :]
#             return batch_ind, batches
