import os
import subprocess
import pathlib
import re
import ast
import pickle


class RunModel2:
    """
    Run a computational model at specified sample points.


    """

    def __init__(self, samples=None, ntasks=1, model_script=None, input_template=None, var_names=None,
                 model_class_name=None, output_script=None, output_class_name=None, cores=1):
        """

        :param samples:
        :param ntasks:
        :param model_script:
        :param input_template:
        :param var_names:
        :param model_class_name:
        :param output_script:
        :param output_class_name:
        :param cores:
        """

        # Check if samples are provided
        if samples is None:
            raise ValueError('Samples must be provided as input to RunModel.')
        else:
            self.samples = samples
            self.nsim = self.samples.shape[0]  # This assumes that the number of rows is the number of simulations.

        # number of cores
        self.ntasks = ntasks

        # number of jobs
        self.cores = cores

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
        # Call the input function
        self._input()

        ################################################################################################################
        # Model evaluation
        # Call the model
        self._execute()

        ################################################################################################################
        # Output handling
        self.output_script = output_script
        self.output_class_name = output_class_name  # TODO: Check if this is necessary or not

        # self.qoi_dict = {}
        # Initialize a list of nsim empty lists. The ith empty list will hold the qoi of the ith simulation.
        self.qoi_list = [[] for i in range(self.nsim)]
        self._output()

    ####################################################################################################################
    # Functions related to running the model - generate input, run the model, and collect the output
    def _input(self):
        if self.input_template is not None:

            print('Here - template input file present!')

            # Check if var_names is a list of strings
            if self._is_list_of_strings(self.var_names):
                # self.var_names = self.var_names
                self.n_vars = len(self.var_names)
            else:
                raise ValueError("Variable names should be passed as a list of strings.")

            # TODO: Check if it is a file
            with open(self.input_template, 'r') as f:
                self.template_text = str(f.read())

            # Loop over the number of samples and create input files in a folder in current directory
            print()
            for i in range(self.nsim):
                # print('The samples being sent are: ' + str(self.samples[i]))

                # Create new text to write to file
                self.new_text = self._find_and_replace_var_names_with_values(var_names=self.var_names,
                                                                             samples=self.samples[i],
                                                                             template_text=self.template_text,
                                                                             index=i,
                                                                             user_format='{:.4E}')
                # Write the new text to the input file
                self._create_input_files(file_name=self.input_template, num=i + 1, text=self.new_text,
                                         new_folder='InputFiles')
            print('Created ' + str(self.nsim) + ' input files in the directory ./InputFiles. \n')

                # TODO: Add variable n_proc which uses the number of cpus and decides the number of jobs

        else:  # If there is no template input file

            print('Here - no template input file!')

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

            else:  # If there is at least one class in the module
                # If the model class name is not given as input and there is only one class, take that class name to
                # run the model.
                if self.model_class_name is None and len(class_list) == 1:
                    self.model_class_name = class_list[0]

                # If there is a model_class_name given, check if it is in the list.
                if self.model_class_name in class_list:
                    print('The model that will be run: ' + self.model_class_name)

                else:
                    print('You specified the model class name as: ' + str(self.model_class_name))
                    raise ValueError("The class name should be specified in the inputs.")

    # TODO: Create folders for each model evaluation

    ####################################################################################################################
    def _execute(self):
        if self.input_template is not None:
            # Create the command to run the model
            # Check if parallel processing is necessary or not. If not, build a command string without gnu parallel.
            if self.ntasks == 1:
                for i in range(self.nsim):
                    # self.model_command = (["python3", str(self.model_script), str(i)])
                    self.model_command = ("python3 " + str(self.model_script) + ' ' + str(i + 1))
                    os.system(self.model_command)
                    # subprocess.run(self.model_command)
            else:
                # self.model_command = ("parallel -j " + str(self.ntasks) + " 'python3 -u " + str(self.model_script) +
                #                       " {1} ::: {1.." + str(self.nsim) + "}' ")
                self.parallel_string = "parallel --delay 0.2 --joblog logs/runtask.log --resume -j " + str(self.ntasks)
                #
                # TODO: Add features for execution on MARCC (SLURM commands)
                # self.srun_string = "srun "

                self.model_command = ([self.parallel_string, " 'python3 -u " + str(self.model_script) +
                                      " {1} ::: {1.." + str(self.nsim) + "}'"])
        else:
            # TODO: Get the signature of the class to get the arguments.
            if self.ntasks == 1:
                # self.model_command = ("python3" + "python_model." + str(self.model_class_name) + "(" + str(self.samples) +
                #                       ")")
                # print("python " + self.model_script + " --samples='" + str(self.samples) + "'")
                # print(["python", self.model_script,   '--samples=', str(self.samples)])
                # self.model_command = (["python", self.model_script,  '--samples=', str(self.samples)])
                self.model_command = ("python " + self.model_script + " --samples='" + str(self.samples) + "'")
            else:
                self.model_command = ("parallel -j " + str(self.ntasks) + " 'python3 python_model." +
                                      str(self.model_class_name) + "(" + str(self.samples) + ")'")

            subprocess.run(self.model_command)

        # print('The command passed to subprocess.run() is:')
        # print(self.model_command)

        # command = "parallel 'python3 -u " + str(self.model_name) + " {1} ::: {1.." + str(self.nsim) + "}"
        # os.system(self.model_command)
        # subprocess.run(self.model_command)

    ####################################################################################################################
    def _output(self):
        # TODO: Check if output class is necessary. If not, get rid of the output class and get output from module.
        if self.input_template is not None:
            if self.ntasks == 1:
                for i in range(self.nsim):

                    self.output_command = 'python3 ' + self.output_script + ' ' + str(i+1)
                    qoi_temp = ast.literal_eval(subprocess.getoutput(self.output_command))
                    # print(qoi_temp)
                    # print(type(qoi_temp))
                    self._collect_output(self.qoi_list, qoi_temp, i)
            # Create the command to process the output
        #     self.output_command = ("parallel -j " + str(self.ntasks) + " 'python3 -u " + str(self.output_script) +
        #                            " {1} ::: {1.." + str(self.nsim) + "}' ")
        else:
            self.output_command = ("parallel -j " + str(self.ntasks) + " 'python3 python_model." +
                                   str(self.output_class_name) + "(" + str(self.samples) + ")'")

        # subprocess.run(self.output_command)
        # TODO: Check if qoi output can be captured by subprocess cleanly without the output printed to screen.
        # self._collect_output(self.qoi_list, qoi_output, pos)
        print(self.qoi_list)
        qoi_list = self.qoi_list
        with open('qoi_output.pkl', 'wb') as f:
            pickle.dump(qoi_list, f)
        # pickle.dump(qoi_list, open()

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

    def _collect_output(self, qoi_list, qoi_output, pos):
        qoi_list[pos] = qoi_output
        return qoi_list

# if __name__ == "__main__":
#     RunModel2._input()
