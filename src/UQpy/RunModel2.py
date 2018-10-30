import os
import subprocess
import pathlib
import re


class RunModel2:
    """
    Run a computational model at specified sample points.


    """

    def __init__(self, samples=None, ncores=1, model_script=None, input_template=None, var_names=None,
                 model_class_name=None, output_script=None, output_class_name=None, ncpus=1):
        """

        :param samples:
        :param ncores:
        :param model_script:
        :param input_template:
        :param var_names:
        :param model_class_name:
        :param output_script:
        :param output_class_name:
        :param ncpus:
        """

        # Check if samples are provided
        if samples is None:
            raise ValueError('Samples must be provided as input to RunModel.')
        else:
            self.samples = samples
            self.nsim = self.samples.shape[0]  # This assumes that the number of rows is the number of simulations.

        # number of cores
        self.ncores = ncores

        # number of jobs
        self.ncpus = ncpus

        # Check if the model script is a python script
        model_extension = pathlib.Path(model_script).suffix
        if model_extension == '.py':
            self.model_script = model_script
        else:
            raise ValueError("The model script must be the name of a python script, with extension '.py'.")

        # Save the model class name
        self.model_class_name = model_class_name

        ################################################################################################################
        # Input handling
        # Read the input template and save the text as a string
        self.input_template = input_template
        self.var_names = var_names

        ################################################################################################################
        # Output handling
        self.output_script = output_script
        self.output_class_name = output_class_name  # TODO: Check if this is necessary or not

        # self.qoi_dict = {}
        # Initialize a list of nsim empty lists. The ith empty list will hold the qoi of the ith simulation.
        self.qoi_list = [[] for i in range(self.nsim)]

    ####################################################################################################################
    # The actual call to the model - generates input, runs the model, and collects the output
    def _input(self):
        if self.input_template is not None:
            # Check if var_names is a list of strings
            if self._is_list_of_strings(self.var_names):
                # self.var_names = self.var_names
                self.n_vars = self.var_names.shape[0]
            else:
                raise ValueError("Variable names should be passed as a list of strings.")

            # TODO: Check if it is a file
            with open(self.input_template, 'r') as f:
                self.template_text = str(f.read())

            # Loop over the number of samples and create input files in a folder in current directory
            for i in range(len(self.nsim)):
                # Create new text to write to file
                self.new_text = self._find_and_replace_var_names_with_values(var_names=self.var_names,
                                                                             samples=self.samples,
                                                                             template_text=self.template_text,
                                                                             index=i,
                                                                             user_format='{:.4E}')
                # Write the new text to the input file
                self._create_input_files(file_name=self.input_template, num=i + 1, text=self.new_text,
                                         new_folder='InputFiles')

                # Create the command to run the model
                # TODO: Check if parallel processing is necessary or not. If not, build a different command string.
                self.model_command = ("parallel -j" + str(self.ncores) + "'python3 -u " + str(self.model_script) +
                                      " {1} ::: {1.." + str(self.nsim) + "}' ")

                # Create the command to process the output
                self.output_command = ("parallel -j" + str(self.ncores) + "'python3 -u " + str(self.output_script) +
                                       " {1} ::: {1.." + str(self.nsim) + "}' ")
                # TODO: Add variable n_proc which uses the number of cpus and decides the number of jobs

        else:  # If there is no template input file
            # Import the python module
            # TODO: Check if model_script is full path or not. Make sure it works in both cases.
            self.python_model = __import__(self.model_script[:-3])
            # Get the names of the classes in the imported module
            import inspect
            class_list = []
            for name, obj in inspect.getmembers(self.python_model):
                if inspect.isclass(obj):
                    class_list.append(name)
            # There should be at least one class in the module - if not there, exit with error.
            if class_list is []:
                raise ValueError("The python model should be defined as a class. Refer documentation for details.")
            else:
                # If there is a model_class_name given, check if it is in the list.
                if self.model_class_name in class_list:
                    # TODO: Get the signature of the class to get the arguments.
                    self.model_command = ("parallel -j" + str(self.ncores) + "'python3 python_model." +
                                          str(self.model_class_name) + "(" + str(self.samples) + ")'")
                    self.output_command = ("parallel -j" + str(self.ncores) + "'python3 python_model." +
                                           str(self.output_class_name) + "(" + str(self.samples) + ")'")
                    # TODO: Check if output class is necessary. If not, get rid of the output class and get output from
                    # module.
                else:
                    raise ValueError("The class name should be correctly specified in the inputs.")

    # TODO: Create folders for each model evaluation

    def _execute(self):
        # command = "parallel 'python3 -u " + str(self.model_name) + " {1} ::: {1.." + str(self.nsim) + "}"
        # os.system(command)
        subprocess.call(self.model_command)

    def _output(self):
        subprocess.call(self.output_command)
        # TODO: Check if qoi output can be captured by subprocess cleanly without the output printed to screen.
        self._collect_output(self.qoi_list, qoi_output, pos)

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
        new_text = ''
        for i in range(len(var_names)):
            string_regex = re.compile(r"<" + var_names[i] + r">")
            count = 0
            for string in string_regex.findall(template_text):
                new_text = template_text[0:template_text.index(string)] + str(user_format.format(float(samples[i]))) \
                           + template_text[(template_text.index(string) + len(string)):]
                count += 1
            if index == 0:
                print("Found: " + str(count) + " instances of word: " + var_names[i])
        return new_text

    def _is_list_of_strings(self, lst):
        return bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)

    def _collect_output(self, qoi_list, qoi_output, pos):
        qoi_list[pos] = list(qoi_output)
        return qoi_list

# if __name__ == "__main__":
#     RunModel2._input()
