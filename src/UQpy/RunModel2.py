import os
import subprocess
import pathlib
import re
import shlex
import datetime


class RunModel2:
    """
    Run a computational model at specified sample points.

    This class is the interface between UQpy and models. If the model is in python, UQpy can interface with the model
    without the need for an input file. In this case, UQpy imports the model module and executes the model object.


    """

    def __init__(self, samples=None, model_script=None, model_object_name=None,
                 input_template=None, var_names=None, output_script=None, output_object_name=None,
                 ntasks=1, cores_per_task=1, nodes=1, resume=False, verbose=True):
        """
        Check input and call appropriate functions in the workflow depending on ntasks and input_template

        :param samples: Samples to be passed as inputs to the model
        :param model_script: The filename of the script which contains commands to execute the model
        :param model_object_name: The name of the function or class which executes the model
        :param input_template: The name of the template input file which will be used to generate input files for each
        run of the model. Refer documentation for more details.
        :param var_names: A list containing the names of the variables which are present in the template input files
        :param output_script: The filename of the script which contains the commands to process the output
        :param output_object_name: The name of the function or class which has the output values. If the object is a
        class, the output must be saved as self.qoi
        :param ntasks: Number of tasks to be run in parallel
        :param cores_per_task: Number of cores to be used by each task
        :param nodes: On MARCC, each node has 24 cores_per_task. Specify the number of nodes if more than one node is
        required.
        """

        # Check if samples are provided
        if samples is None:
            raise ValueError('Samples must be provided as input to RunModel.')
        else:
            self.samples = samples
            self.nsim = len(self.samples)  # This assumes that the number of rows is the number of simulations.

            # TODO: Make it clear if numpy arrays or lists can be passed as samples
            # TODO: Raise a warning if the shape is not defined clearly
            # TODO: Check if fire is installed
            # TODO: Support verbose options - print messages only if verbose is true
        # Input related
        self.input_template = input_template
        self.var_names = var_names
        # Check if var_names is a list of strings
        if self._is_list_of_strings(self.var_names):
            self.n_vars = len(self.var_names)
        else:
            raise ValueError("Variable names should be passed as a list of strings.")

        # Model related
        # TODO: Check if model_script is full path or not. Make sure it works in both cases.
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
        # Initialize a list of nsim empty lists. The ith empty list will hold the qoi of the ith simulation.
        self.qoi_list = [[] for i in range(self.nsim)]

        # Number of tasks
        self.ntasks = ntasks
        # Number of cores_per_task
        self.cores_per_task = cores_per_task
        # Number of nodes
        self.nodes = nodes

        # Check if there is a template input file or not and execute the appropriate function
        if self.input_template is not None:  # If there is a template input file
            # Check if it is a file and is readable
            assert os.path.isfile(self.input_template) and os.access(self.input_template, os.R_OK), \
                "File {} doesn't exist or isn't readable".format(self.input_template)
            # Read in the text from the template file
            with open(self.input_template, 'r') as f:
                self.template_text = str(f.read())

            # Import the output script
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

    ####################################################################################################################
    def _serial_execution(self):
        """
        Perform serial execution of the model when there is a template input file
        :return:
        """
        print('\nPerforming serial execution of the model with template input.\n')
        # Loop over the number of simulations, executing the model once per loop
        for i in range(self.nsim):
            # Call the input function
            self._input_serial(i)

            # Execute the model
            self._execute_serial(i)

            # Call the output function
            self._output_serial(i)

    ####################################################################################################################
    def _parallel_execution(self):
        """
        Execute the model in parallel when there is a template input file
        :return:
        """
        print('\nPerforming parallel execution of the model with template input.\n')
        # Call the input function
        print('\nCreating inputs in parallel execution of the model with template input.\n')
        self._input_parallel()

        # Execute the model
        print('\nExecuting the model in parallel with template input.\n')
        self._execute_parallel()

        # Call the output function
        print('\nCollecting outputs in parallel execution of the model with template input.\n')
        # TODO: Make the output collection execute in parallel if possible
        for i in range(self.nsim):
            self._output_parallel(i)

    ####################################################################################################################
    def _serial_python_execution(self):
        """
        Execute the python model in serial when there is no template input file
        :return:
        """
        print('\nPerforming serial execution of the model without template input.\n')
        # Run python model
        for i in range(self.nsim):
            exec('from ' + self.model_script[:-3] + ' import ' + self.model_object_name)
            self.model_output = eval(self.model_object_name + '(self.samples[i])')
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
        print('\nPerforming parallel execution of the model without template input.\n')
        # TODO: Run python model in parallel using multiprocess
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

    def _run_parallel_python_chunked(self, some_samples):
        par_res = [[] for i in range(some_samples.shape[0])]
        for i in range(some_samples.shape[0]):
            exec('from ' + self.model_script[:-3] + ' import ' + self.model_object_name)
            parallel_output = eval(self.model_object_name + '(some_samples[i])')
            if self.model_is_class:
                par_res[i] = parallel_output.qoi
            else:
                par_res[i] = parallel_output

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
        self._create_input_files(file_name=self.input_template, num=index + 1, text=self.new_text,
                                 new_folder='InputFiles')

    def _execute_serial(self, index):
        """
        Execute the model once using the input file of index number
        :param index: The simulation number
        :return:
        """
        # TODO: Get the alias to the command to run python as optional input from user
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

    def _input_parallel(self):
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
            # Write the new text to the input file
            self._create_input_files(file_name=self.input_template, num=i + 1, text=new_text,
                                     new_folder='InputFiles')
        print('Created ' + str(self.nsim) + ' input files in the directory ./InputFiles. \n')

    def _execute_parallel(self):
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
        self.model_command_string = (self.parallel_string + " 'python3 -u " + str(self.model_script) +
                                     "' {1} ::: {0.." + str(self.nsim - 1) + "}")
        # print(self.model_command_string)

        # self.model_command = shlex.split(self.model_command_string)
        # TODO: Identify why using shlex does not work and get rid of shell=True if possible
        # print(self.model_command)

        # self.model_command = (["parallel", "--delay", "0.2", "--joblog", "logs/runtask.log",
        #                       "--resume", "-j", str(self.ntasks), "'python3 -u " + str(self.model_script) +
        #                        "'", "{1} ::: {1.."+str(self.nsim)+"}"])
        # print(self.model_command)

        # subprocess.run(self.model_command)
        subprocess.run(self.model_command_string, shell=True)

    def _output_parallel(self, index):
        """
        Extract output from parallel execution
        :param index: The simulation number
        :return:
        """
        # TODO: Make this run in parallel if possible
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
        new_text = template_text
        for j in range(len(var_names)):
            string_regex = re.compile(r"<" + var_names[j] + r">")
            count = 0
            for string in string_regex.findall(template_text):
                new_text = new_text[0:new_text.index(string)] + str(user_format.format(float(samples[j]))) \
                           + new_text[(new_text.index(string) + len(string)):]
                count += 1
            if index == 0:
                if count > 1:
                    print("Found " + str(count) + " instances of variable: '" + var_names[j] + "' in the input file.")
                else:
                    print("Found " + str(count) + " instance of variable: '" + var_names[j] + "' in the input file.")
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
                print('The model class that will be run: ' + self.model_object_name)
                self.model_is_class = True
            elif self.model_object_name in function_list:
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
                print('The output class that will be run: ' + self.output_object_name)
                self.output_is_class = True
            elif self.output_object_name in function_list:
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

    # ####################################################################################################################
    # # Functions related to running the model - generate input, run the model, and collect the output
    # def _input(self):
    #     if self.input_template is not None:
    #
    #         print('Here - template input file present!')
    #
    #         # Check if var_names is a list of strings
    #         if self._is_list_of_strings(self.var_names):
    #             # self.var_names = self.var_names
    #             self.n_vars = len(self.var_names)
    #         else:
    #             raise ValueError("Variable names should be passed as a list of strings.")
    #
    #         with open(self.input_template, 'r') as f:
    #             self.template_text = str(f.read())
    #
    #         # Loop over the number of samples and create input files in a folder in current directory
    #         print()
    #         for i in range(self.nsim):
    #             # print('The samples being sent are: ' + str(self.samples[i]))
    #
    #             # Create new text to write to file
    #             self.new_text = self._find_and_replace_var_names_with_values(var_names=self.var_names,
    #                                                                          samples=self.samples[i],
    #                                                                          template_text=self.template_text,
    #                                                                          index=i,
    #                                                                          user_format='{:.4E}')
    #             # Write the new text to the input file
    #             self._create_input_files(file_name=self.input_template, num=i + 1, text=self.new_text,
    #                                      new_folder='InputFiles')
    #         print('Created ' + str(self.nsim) + ' input files in the directory ./InputFiles. \n')
    #
    #
    #     else:  # If there is no template input file
    #
    #         print('\nHere - no template input file!\n')
    #
    #         # Import the python module
    #         self.python_model = __import__(self.model_script[:-3])
    #         # Get the names of the classes in the imported module
    #         import inspect
    #         class_list = []
    #         for name, obj in inspect.getmembers(self.python_model):
    #             if inspect.isclass(obj):
    #                 class_list.append(name)
    #
    #         # There should be at least one class in the module - if not there, exit with error.
    #         if class_list is []:
    #             raise ValueError("The python model should be defined as a class. Refer documentation for details.")
    #
    #         else:  # If there is at least one class in the module
    #             # If the model class name is not given as input and there is only one class, take that class name to
    #             # run the model.
    #             if self.model_object_name is None and len(class_list) == 1:
    #                 self.model_object_name = class_list[0]
    #
    #             # If there is a model_object_name given, check if it is in the list.
    #             if self.model_object_name in class_list:
    #                 print('The model that will be run: ' + self.model_object_name)
    #
    #             else:
    #                 print('You specified the model class name as: ' + str(self.model_object_name))
    #                 raise ValueError("The class name should be specified in the inputs.")
    #
    # # TODO: Create folders for each model evaluation
    #
    # ####################################################################################################################
    # def _execute(self):
    #     if self.input_template is not None:
    #         # Create the command to run the model
    #         # Check if parallel processing is necessary or not. If not, build a command string without gnu parallel.
    #         if self.ntasks == 1:
    #             for i in range(self.nsim):
    #                 # self.model_command = (["python3", str(self.model_script), str(i)])
    #                 # subprocess.run(self.model_command)
    #                 self.model_command = ("python3 " + str(self.model_script) + ' ' + str(i + 1))
    #                 os.system(self.model_command)
    #         else:
    #             # self.model_command = ("parallel -j " + str(self.ntasks) + " 'python3 -u " + str(self.model_script) +
    #             #                       " {1} ::: {1.." + str(self.nsim) + "}' ")
    #             self.parallel_string = "parallel --delay 0.2 --joblog logs/runtask.log --resume -j " + str(
    #                 self.ntasks)
    #             #
    #             # TODO: Add features for execution on MARCC (SLURM commands)
    #             # self.srun_string = "srun "
    #
    #             self.model_command = ([self.parallel_string, " 'python3 -u " + str(self.model_script) +
    #                                    " {1} ::: {1.." + str(self.nsim) + "}'"])
    #     else:
    #         if self.ntasks == 1:
    #
    #             exec('from ' + self.model_script[:-3] + ' import ' + self.model_object_name)
    #             self.model_output = eval(self.model_object_name + '(self.samples)')
    #
    #             # print("\nThe model output is:")
    #             # print(self.model_output.model_output)
    #             #
    #             # print("\nThe qoi is:")
    #             # print(self.model_output.qoi)
    #
    #         else:
    #             self.model_command = ("parallel -j " + str(self.ntasks) + " 'python3 python_model." +
    #                                   str(self.model_object_name) + "(" + str(self.samples) + ")'")
    #
    #     # Run the model
    #     # subprocess.run(self.model_command)
    #     # out_temp = subprocess.getoutput(self.model_command)
    #     # out_temp = ast.literal_eval(subprocess.getoutput(self.model_command))
    #     # print(out_temp)
    #     # print(type(out_temp))
    #
    #     # print('The command passed to subprocess.run() is:')
    #     # print(self.model_command)
    #
    #     # command = "parallel 'python3 -u " + str(self.model_name) + " {1} ::: {1.." + str(self.nsim) + "}"
    #     # os.system(self.model_command)
    #     # subprocess.run(self.model_command)
    #
    # ####################################################################################################################
    # def _output(self):
    #
    #     if self.input_template is not None:
    #         if self.ntasks == 1:
    #             for i in range(self.nsim):
    #                 self.output_command = 'python3 ' + self.output_script + ' ' + str(i + 1)
    #                 qoi_temp = ast.literal_eval(subprocess.getoutput(self.output_command))
    #                 print(qoi_temp)
    #                 print(type(qoi_temp))
    #                 self._collect_output(self.qoi_list, qoi_temp, i)
    #         # Create the command to process the output
    #     #     self.output_command = ("parallel -j " + str(self.ntasks) + " 'python3 -u " + str(self.output_script) +
    #     #                            " {1} ::: {1.." + str(self.nsim) + "}' ")
    #     else:
    #         self.output_command = ("parallel -j " + str(self.ntasks) + " 'python3 python_model." +
    #                                str(self.output_object_name) + "(" + str(self.samples) + ")'")
    #
    #     # subprocess.run(self.output_command)
    #
    #     # self._collect_output(self.qoi_list, qoi_output, pos)
    #     print()
    #     # print(self.qoi_list)

# if __name__ == "__main__":
#     RunModel2._input()
