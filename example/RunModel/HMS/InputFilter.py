import collections
import os
import re
import sys
from shutil import copyfile

import hms
import numpy as np


class PickleableSWIG:
    """
    Allow SWIG type objects to be pickled
    """

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        state = self.__dict__
        if "this" in state:
            del state["this"]
        return state


class Argument(hms.Argument, PickleableSWIG):

    def __init__(self, nvoids, samples, index):
        # Argument must have a variable 'index' that is used to keep track of run numbers

        self.x = [None]*nvoids
        self.y = [None]*nvoids
        self.z = [None]*nvoids
        self.R = [None]*nvoids
 
        print(nvoids)
        print(samples)

        samples = np.reshape(np.atleast_2d(samples), (nvoids, 4))

        print(samples)

        hms.Argument.__init__(self)
        for i in range(nvoids):
            self.x[i] = samples[i, 0]
            self.y[i] = samples[i, 1]
            self.z[i] = samples[i, 2]
            self.R[i] = samples[i, 3]

        self.index = index

        print(os.getcwd())

        return


class InputFilter(hms.InputFilter, PickleableSWIG):
    """
    This is a generic Input Filter for HMS models. It functions as an input modifier to create unique input files for
    each model evaluation from a template input file.

    * **hms_point_file_name** ('str')
        This is the template input file. It follows all the same formatting as the ``RunModel'' input template file
        specified by 'input_template'. See the ``RunModel'' class.

    * **var_names** (`list` of `str`)
        A list containing the names of the variables present in `input_template`.

        If `input template` is provided and  `var_names` is not passed, i.e. if ``var_names=None``, then the default
        variable names `x0`, `x1`, `x2`,..., `xn` are created and used by ``RunModel``, where `n` is the number of
        variables (`n_vars`).

        The number of variables is equal to the second dimension of `samples` (i.e. ``n_vars=len(samples[0])``).

    * **files2copy** (`list` of `str`)
        A list containing paths to any additional files that must be copied to the local run directory.

    * **fmt** (`str`)
        If the `template_input` requires variables to be written in specific format, this format can be specified here.

        Format specification follows standard Python conventions for the str.format() command described at:
        https://docs.python.org/3/library/stdtypes.html#str.format. For additional details, see the Format String Syntax
        description at: https://docs.python.org/3/library/string.html#formatstrings.

        For example, ls-dyna .k files require each card is to be exactly 10 characters. The following format string
        syntax can be used, "{:>10.4f}".

    * **separator** (`str`)
        A string used to delimit values when printing arrays to the `template_input`.

        `separator` is not used in the Python model workflow.
    """


    def __init__(self, hms_point_file_name=None, var_names=None, files2copy=None, fmt=None, separator=', '):

        hms.InputFilter.__init__(self)

        self.pointFileName = hms_point_file_name
        if self.pointFileName is None:
            raise TypeError('\nUQpy: The user must specify an HMS point file.\n')
        self.template_text = ''
        self.n_vars = None
        self.samples = []

        self.var_names = var_names
        self.fmt = fmt
        self.separator = separator
        self.files2copy=files2copy

        self.path = None
        self.argument = None

        return

    def apply(self, argument=None, directory=None):

        self.argument = argument
        if self.argument is None:
            raise TypeError('\nUQpy: The user must specify an HMS argument.\n')

        # Set the path to the run directory
        self.path = os.path.join(directory, self.pointFileName)

        # Set the path to the template input file
        template_path = os.path.join(os.getcwd(), self.pointFileName)

        # Copy the template input file to the run directory
        copyfile(template_path, self.path)

        # Copy additional files
        for f in self.files2copy:
            copy_from = os.path.join(os.getcwd(), f)
            copy_to = os.path.join(directory, f)
            copyfile(copy_from, copy_to)
            sys.stderr.write('Copying '+ copyfrom+ ' to '+ copyto + '\n')

        if hasattr(self, 'var_names'):
            self.n_vars = len(self.var_names)
        else:
            self.n_vars = len(self.samples)
            self.var_names = []
            for i in range(self.n_vars):
                self.var_names.append('x%d' % i)

        counter = 0

        for attr, value in self.argument.__dict__.items():
            if attr != 'index':
                counter += counter
                self.samples.append(np.asarray(value))

        # Check whether the template input file exists.
        if self.pointFileName is not None:
            # Check if it is a file and is readable
            assert os.path.isfile(self.path) and os.access(self.path, os.R_OK), \
                "\nUQpy: File {} doesn't exist or isn't readable".format(self.pointFileName)

            # Read in the text from the template files
            with open(template_path, 'r') as f:
                self.template_text = str(f.read())
        # sys.stderr('Testing')

        # Replace variable names with values.
        new_text = self._find_and_replace_var_names_with_values()

        # Create each individual input file.
        self._create_input_files(file_name=self.path, text=new_text)


    def _find_and_replace_var_names_with_values(self):
        """
        Replace placeholders containing variable names in template input text with sample values.
        """

        sys.stderr.write('Testing')

        template_text = self.template_text
        var_names = self.var_names
        samples = self.samples

        new_text = template_text
        for j in range(self.n_vars):
            string_regex = re.compile(r"<" + var_names[j] + r".*?>")

            sys.stderr.write('Testing')
            count = 0
            for string in string_regex.findall(template_text):
                temp_check = string[1:-1].split("[")[0]
                pattern_check = re.compile(var_names[j])
                if pattern_check.fullmatch(temp_check):
                    temp = string[1:-1].replace(var_names[j], "samples[" + str(j) + "]")
                    sys.stderr.write(str(temp))
                    try:
                        temp = eval(temp)
                    except IndexError as err:
                        sys.stderr.write('IndexError')
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
        return new_text

    @staticmethod
    def _create_input_files(file_name, text):
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

        with open(file_name, 'w') as f:
            f.write(text)
        return

class LSOutputFilter(hms.OutputFilter, PickleableSWIG):

    def __init__(self):
        hms.OutputFilter.__init__(self)
        return

    def apply(self, directory, stdOut, argument):
        f = open(stdOut)
        value = float(f.readline())
        return LSValue(value)


class LSValue(hms.Value, PickleableSWIG):

    def __init__(self, value):
        hms.Value.__init__(self)
        self.value = value
        return
