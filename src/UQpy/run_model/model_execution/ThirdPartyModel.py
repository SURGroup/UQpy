import collections
import datetime
import logging
import os
import pathlib
import platform
import re
import shutil
import subprocess

import numpy as np


class ThirdPartyModel:

    def __init__(self, var_names: list[str], input_template: str, model_script: str, output_script: str = None,
                 model_object_name: str = None, output_object_name: str = None, fmt: str = None, separator: str = ', ',
                 delete_files: bool = False, model_dir: str = "Model_Runs"):
        """

        :param var_names: A list containing the names of the variables present in `input_template`.

         If `input template` is provided and  `var_names` is not passed, i.e. if ``var_names=None``, then the default
         variable names `x0`, `x1`, `x2`,..., `xn` are created and used by :class:`.RunModel`, where `n` is the number of
         variables (`n_vars`).

         The number of variables is equal to the second dimension of `samples` (i.e. ``n_vars=len(samples[0])``).

         `var_names` is not used in the Python model workflow.
        :param input_template: The name of the template input file that will be used to generate input files for
         each run of the model. When operating :class:`.RunModel` with a third-party software model, ``input_template`` must
         be specified.

         The named file must be present in the current working directory from which :class:`.RunModel` is called.
        :param model_script: The filename (with .py extension) of the Python script which contains commands to
         execute the model.

         The named file must be present in the current working directory from which :class:`.RunModel` is called.
        :param model_object_name: In the Python workflow, `model_object_name` specifies the name of the function or
         class within `model_script' that executes the model. If there is only one function or class in the
         `model_script`, then it is not necessary to specify the model_object_name. If there are multiple objects within
         the `model_script`, then `model_object_name` must be specified.
        :param output_script: The filename of the Python script that contains the commands to process the output from
         third-party software model evaluation. `output_script` is used to extract quantities of interest from model
         output files and return the quantities of interest to :class:`.RunModel` for subsequent :py:mod:`UQpy` processing (e.g. for
         adaptive methods that utilize the results of previous simulations to initialize new simulations).

         If, in the third-party software model workflow, ``output_script = None`` (the default), then the
         :py:attr:`qoi_list` attribute is empty and postprocessing must be handled outside of :py:mod:`UQpy`.

         If used, the named file must be present in the current working directory from which :class:`.RunModel` is called.

         `output_script` is not used in the Python model workflow. In the Python model workflow, all model postprocessing
         is handled directly within `model_script`.
        :param output_object_name: The name of the function or class within `output_script` that is used to collect
         and process the output values from third-party software model output files. If the object is a class, the
         output must be saved as an attribute called :py:attr:`qoi`. If it is a function, it should return the output
         quantity of interest.

         If there is only one function or only one class in `output_script`, then it is not necessary to specify
         `output_object_name`. If there are multiple objects in `output_script`, then output_object_name must be
         specified.
        :param fmt: If the `template_input` requires variables to be written in specific format, this format can be
         specified here.

         Format specification follows standard Python conventions for the str.format() command described at:
         https://docs.python.org/3/library/stdtypes.html#str.format. For additional details, see the Format String Syntax
         description at: https://docs.python.org/3/library/string.html#formatstrings.

         For example, ls-dyna .k files require each card is to be exactly 10 characters. The following format string
         syntax can be used, "{:>10.4f}".

        :param separator: A string used to delimit values when printing arrays to the `template_input`.
        :param delete_files: Specifies whether or not to delete individual run output files after model execution
         and output processing.

         If `delete_files = True`, :class:`.RunModel` will remove all `run_i...` directories in the `model_dir`.
        :param model_dir: Specifies the name of the sub-directory from which the model will be executed and to which
         output files will be saved.  A new directory is created by :class:`.RunModel` within the current directory whose name
         is `model_dir` appended with a timestamp.
        """
        self.template_text = None
        self.logger = logging.getLogger(__name__)

        if platform.system() in ["Windows"]:
            self.python_command = "python"
        else:
            self.python_command = "python3"

        self.separator = separator
        self.fmt = fmt
        self.check_formatting(fmt)
        self.delete_files = delete_files

        self.input_template = input_template
        self.var_names = var_names
        self.n_variables: int = 0

        if self.var_names is not None and not ThirdPartyModel._is_list_of_strings(self.var_names):
            raise ValueError("\nUQpy: Variable names should be passed as a list of strings.\n")

        # Establish parent directory for simulations
        self.parent_dir = os.getcwd()

        # Create a list of all of the files and directories in the working directory. Do not include any other
        # directories containing the same name as model_dir
        model_files = self.create_model_files_list(model_dir)

        # Create a new directory where the model will be executed
        self.create_model_execution_directory(model_dir, model_files)

        # Check if the model script is a python script
        model_extension = pathlib.Path(model_script).suffix
        if model_extension == '.py':
            self.model_script = model_script
        else:
            raise ValueError("\nUQpy: The model script must be the name of a python script, with extension '.py'.")

        self.model_object_name = model_object_name
        self.output_script = output_script
        self.output_object_name = output_object_name

        self._check_python_model()

        # Import the output script
        if self.output_script is not None:
            # self.output_module = __import__(self.output_script[:-3])
            # Run function which checks if the output module has the output object
            self._check_output_module()

    def create_model_execution_directory(self, model_dir, model_files):
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%f")
        self.model_dir = os.path.join(self.parent_dir, model_dir + "_" + ts)
        os.makedirs(self.model_dir)

        os.chdir(self.model_dir)

        self.logger.info("\nUQpy: The following directory has been created for model evaluations: \n" + self.model_dir)
        # Copy files from the model list to model run directory
        for file_name in model_files:
            full_file_name = os.path.join(self.parent_dir, file_name)
            if not os.path.isdir(full_file_name):
                shutil.copy(full_file_name, self.model_dir)
            else:
                new_dir_name = os.path.join(self.model_dir, os.path.basename(full_file_name))
                shutil.copytree(full_file_name, new_dir_name)
        self.logger.info("\nUQpy: The model files have been copied to the following directory for evaluation: \n"
                         + self.model_dir)
        parent_dir = os.path.dirname(self.model_dir)
        os.chdir(parent_dir)

    def create_model_files_list(self, model_dir):
        model_files = []
        for f_name in os.listdir(self.parent_dir):
            path = os.path.join(self.parent_dir, f_name)
            if model_dir not in path:
                model_files.append(path)
        self.model_files = model_files
        return model_files

    def check_formatting(self, fmt):
        if self.fmt is None:
            pass
        elif isinstance(self.fmt, str):
            if (self.fmt[0] != "{") or (self.fmt[-1] != "}") or (":" not in self.fmt):
                raise ValueError("\nUQpy: fmt should be a string in brackets indicating a standard Python format.\n")
        else:
            raise TypeError("\nUQpy: fmt should be a str.\n")

    @staticmethod
    def _is_list_of_strings(list_of_strings):
        """
        Check if input list contains only strings

        ** Input: **

        :param list_of_strings: A list whose entries should be checked to see if they are strings
        :type list_of_strings: list
        """
        return (bool(list_of_strings) and isinstance(list_of_strings, list)
                and all(isinstance(element, str) for element in list_of_strings))

    def initialize(self, samples):
        os.chdir(self.model_dir)
        self.logger.info("\nUQpy: All model evaluations will be executed from the following directory: \n"
                         + self.model_dir)

        self.n_variables = len(samples[0])

        if self.input_template is not None:
            if self.var_names is None:
                # If var_names is not passed and there is an input template, create default variable names
                self.var_names = []
                for i in range(self.n_variables):
                    self.var_names.append('x%d' % i)

            elif len(self.var_names) != self.n_variables:
                raise ValueError("\nUQpy: var_names must have the same length as the number of variables (i.e. "
                                 "len(var_names) = len(samples[0]).\n")
        assert os.path.isfile(self.input_template) and os.access(self.input_template, os.R_OK), \
            "\nUQpy: File {} doesn't exist or isn't readable".format(self.input_template)
        # Read in the text from the template files
        with open(self.input_template, "r") as f:
            self.template_text = str(f.read())

    def finalize(self):
        parent_dir = os.path.dirname(self.model_dir)
        os.chdir(parent_dir)

    def preprocess_single_sample(self, i, sample):
        work_dir = os.path.join(self.model_dir, "run_" + str(i))
        self._copy_files(work_dir=work_dir)

        # Change current working directory to model run directory
        os.chdir(work_dir)
        self.logger.info("\nUQpy: Running model number " + str(i) + " in the following directory: \n" + work_dir)

        # Call the input function
        self._input_serial(i, sample)

    def execute_single_sample(self, index, sample_to_send):
        # os.system(f"{self.python_command} {self.model_script} {index}")
        python_model = __import__(self.model_script[:-3])
        model_object = getattr(python_model, self.model_object_name)
        model_object(index)

    def postprocess_single_file(self, index, model_output):
        if self.output_script is not None:
            output = self._output_serial(index)

        work_dir = os.path.join(self.model_dir, "run_" + str(index))
        # Remove the copied files and folders
        self._remove_copied_files(work_dir)

        # Return to the model directory
        os.chdir(self.model_dir)
        self.logger.info("\nUQpy: Model evaluation " + str(index) + " complete.\n")
        self.logger.info("\nUQpy: Returning to the model directory:\n" + self.model_dir)
        return output

    def _input_serial(self, index, sample):
        """
        Create one input file using the template and attach the index to the filename

        ** Input: **

        :param index: The simulation number
        :type index: int
        """
        self.new_text = self._find_and_replace_var_names_with_values(sample=sample)
        # Write the new text to the input file
        self._create_input_files(file_name=self.input_template, num=index, text=self.new_text,
                                 new_folder="InputFiles", )

    def _create_input_files(self, file_name, num, text, new_folder="InputFiles"):
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
        new_name = os.path.join(
            new_folder, base_name[0] + "_" + str(num) + base_name[1]
        )
        with open(new_name, "w") as f:
            f.write(text)
        return

    def _find_and_replace_var_names_with_values(self, sample):
        """
        Replace placeholders containing variable names in template input text with sample values.

        ** Input: **

        :param index: The sample number
        :type index: int
        """

        template_text = self.template_text
        var_names = self.var_names

        new_text = template_text
        for j in range(self.n_variables):
            string_regex = re.compile(r"<" + var_names[j] + r".*?>")
            count = 0
            for string in string_regex.findall(template_text):
                temp_check = string[1:-1].split("[")[0]
                pattern_check = re.compile(var_names[j])
                if pattern_check.fullmatch(temp_check):
                    temp = string[1:-1].replace(var_names[j], "sample[" + str(j) + "]")
                    try:
                        temp = eval(temp)
                    except IndexError as err:
                        print("\nUQpy: Index Error: {0}\n".format(err))
                        raise IndexError("{0}".format(err))

                    if isinstance(temp, collections.abc.Iterable):
                        # If it is iterable, flatten and write as text file with designated separator
                        temp = np.array(temp).flatten()
                        to_add = ""
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
                    new_text = (new_text[0: new_text.index(string)] + to_add
                                + new_text[(new_text.index(string) + len(string)):])
                    count += 1
        return new_text

    def _output_serial(self, index):
        """
        Execute the output script, obtain the output qoi and save it in qoi_list

        ** Input: **

        :param index: The simulation number
        :type index: int
        """
        # Run output module
        output_module = __import__(self.output_script[:-3])
        output_object = getattr(output_module, self.output_object_name)
        model_output = output_object(index)
        return model_output.qoi if self.model_is_class else model_output

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
        output_module = __import__(self.output_script[:-3])

        class_list = []
        function_list = []
        for name, obj in inspect.getmembers(output_module):
            if inspect.isclass(obj):
                class_list.append(name)
            elif inspect.isfunction(obj):
                function_list.append(name)

        # There should be at least one class or function in the module - if not there, exit with error.
        if len(class_list) == 0 and len(function_list) == 0:
            raise ValueError("\nUQpy: The output object should be defined as a function or class in the script.\n")

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
                # self.logger.info("\nUQpy: The output class that will be run: " + self.output_object_name)
                self.output_is_class = True
            elif self.output_object_name in function_list:
                # self.logger.info("\nUQpy: The output function that will be run: " + self.output_object_name)
                self.output_is_class = False
            else:
                if self.output_object_name is None:
                    raise ValueError("\nUQpy: There are more than one objects in the module. Specify the name of the "
                                     "function or class which has to be executed.\n")
                else:
                    print("\nUQpy: You specified the output object name as: " + str(self.output_object_name))
                    raise ValueError("\nUQpy: The file does not contain an object which was specified as the output "
                                     "processor.\n")

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
        python_model = __import__(self.model_script[:-3])
        for name, obj in inspect.getmembers(python_model):
            if inspect.isclass(obj):
                class_list.append(name)
            elif inspect.isfunction(obj):
                function_list.append(name)

        # There should be at least one class or function in the module - if not there, exit with error.
        if len(class_list) == 0 and len(function_list) == 0:
            raise ValueError("\nUQpy: A python model should be defined as a function or class in the script.\n")

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
                # self.logger.info("\nUQpy: The model class that will be run: " + self.model_object_name)
                self.model_is_class = True
            elif self.model_object_name in function_list:
                # self.logger.info("\nUQpy: The model function that will be run: " + self.model_object_name)
                self.model_is_class = False
            else:
                if self.model_object_name is None:
                    raise ValueError("\nUQpy: There are more than one objects in the module. Specify the name of the "
                                     "function or class which has to be executed.\n")
                else:
                    print("\nUQpy: You specified the model_object_name as: " + str(self.model_object_name))
                    raise ValueError("\nUQpy: The file does not contain an object which was specified as the model.\n")
