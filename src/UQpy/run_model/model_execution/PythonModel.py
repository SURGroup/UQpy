import logging
import pathlib
import platform

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import Numpy2DFloatArray


class PythonModel:
    @beartype
    def __init__(self, model_script: str, model_object_name: str, var_names: list[str] = None,
                 delete_files: bool = False, **model_object_name_kwargs):
        """

        :param model_script: The filename (with .py extension) of the Python script which contains commands to
         execute the model.

         The named file must be present in the current working directory from which :class:`.RunModel` is called.
        :param model_object_name: In the Python workflow, `model_object_name` specifies the name of the function or
         class within `model_script' that executes the model. If there is only one function or class in the
         `model_script`, then it is not necessary to specify the model_object_name. If there are multiple objects within
         the `model_script`, then `model_object_name` must be specified.

         `model_object_name` is not used in the third-party software model workflow.
        :param var_names: A list containing the names of the variables present in `input_template`.

         If `input template` is provided and  `var_names` is not passed, i.e. if ``var_names=None``, then the default
         variable names `x0`, `x1`, `x2`,..., `xn` are created and used by :class:`.RunModel`, where `n` is the number of
         variables (`n_vars`).

         The number of variables is equal to the second dimension of `samples` (i.e. ``n_vars=len(samples[0])``).

         `var_names` is not used in the Python model workflow.
        :param delete_files: Specifies whether or not to delete individual run output files after model execution
         and output processing.

         If `delete_files = True`, :class:`.RunModel` will remove all `run_i...` directories in the `model_dir`.
        :param model_object_name_kwargs: Additional inputs to the Python object specified by `model_object_name` in the
         Python model workflow.
        """
        if var_names is None:
            var_names = []
        self.var_names = var_names
        self._model_output = None
        self.logger = logging.getLogger(__name__)

        if platform.system() in ["Windows"]:
            self.python_command = "python"
        else:
            self.python_command = "python3"

        self.model_object_name = model_object_name
        self.model_object_name_kwargs = model_object_name_kwargs

        self.delete_files = delete_files

        # Check if the model script is a python script
        model_extension = pathlib.Path(model_script).suffix
        if model_extension == ".py":
            self.model_script = model_script
        else:
            raise ValueError("\nUQpy: The model script must be the name of a python script, with extension '.py'.")

        # Import the python module
        python_model = __import__(self.model_script[:-3])
        self.model_object = getattr(python_model, self.model_object_name)
        # Run function which checks if the python model has the model object
        self._check_python_model(python_model)
        self.logger.info('\nUQpy: Performing serial execution of a Python model.\n')

    def initialize(self, samples):
        pass

    def finalize(self):
        pass

    def preprocess_single_sample(self, index, sample) -> Numpy2DFloatArray:
        return np.atleast_2d(sample)

    def execute_single_sample(self, index, sample_to_send):
        if len(self.model_object_name_kwargs) == 0:
            return self.model_object(sample_to_send)
        else:
            return self.model_object(sample_to_send, **self.model_object_name_kwargs)

    def postprocess_single_file(self, index, model_output):
        return model_output.qoi if self.model_is_class else model_output

    def _check_python_model(self, python_model):
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
                self.logger.info("\nUQpy: The model class that will be run: " + self.model_object_name)
                self.model_is_class = True
            elif self.model_object_name in function_list:
                self.logger.info("\nUQpy: The model function that will be run: " + self.model_object_name)
                self.model_is_class = False
            else:
                if self.model_object_name is None:
                    raise ValueError("\nUQpy: There are more than one objects in the module. Specify the name of the "
                                     "function or class which has to be executed.\n")
                else:
                    print("\nUQpy: You specified the model_object_name as: " + str(self.model_object_name))
                    raise ValueError("\nUQpy: The file does not contain an object which was specified as the model.\n")
