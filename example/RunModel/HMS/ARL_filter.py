import hms
import os
import sys
import re
import collections
from shutil import copyfile
import numpy as np
from ARL_utils import *

#
# Allow SWIG type objects to be pickled
#
class PickleableSWIG:

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        state = self.__dict__
        if "this" in state:
            del state["this"]
        return state


class LSArgument(hms.Argument, PickleableSWIG):

    def __init__(self, nvoids, samples, index):
        # Argument must have a variable 'index' that is used to keep track of run numbers

        self.x = [None]*nvoids
        self.y = [None]*nvoids
        self.z = [None]*nvoids
        self.R = [None]*nvoids
 
        print(nvoids)
        print(samples)

        samples = np.reshape(np.atleast_2d(samples), (nvoids, 4))


        hms.Argument.__init__(self)
        for i in range(nvoids):
            self.x[i] = samples[i, 0]
            self.y[i] = samples[i, 1]
            self.z[i] = samples[i, 2]
            self.R[i] = samples[i, 3]

        self.index = index

        return


class LSInputFilter(hms.InputFilter, PickleableSWIG):

    def __init__(self, pointFileName, var_names=None, files2copy=None, fmt=None, separator=', '):
        hms.InputFilter.__init__(self)
        self.pointFileName = pointFileName

        self.template_text = ''
        self.n_vars = None
        self.samples = []

        ################# USER INPUTS ##############################
        # Note that the user is required to specify the variable names, if the default variable names are not used.
        # To use default variable names ['x0', 'x1', ..., 'xn'], comment the following line.
        # self.var_names = ['x', 'y', 'z', 'R']
        self.var_names = var_names

        # Note that the user is required to specify the format for writing variables to the text file. To use default
        # formatting, set self.fmt = None.
        # self.fmt = '{:>10.4f}'
        self.fmt = fmt
        # self.fmt = None

        # Note that the user is required to specify a separator to delineate values when writing iterable variables to
        # the text file. Do not comment the following line.
        # self.separator = ', '
        self.separator = separator

        # Additional files to copy
        # self.files2copy=['shells_refinement[4].k', 'init_vel_ale_3comp.k']
        self.files2copy=files2copy
        ################# END USER INPUTS ##########################

        return

    def apply(self, argument, directory):

        self.argument = argument

        # Set the path to the run directory
        self.path = os.path.join(directory, self.pointFileName)

        # Set the path to the template input file
        template_path = os.path.join(os.getcwd(), self.pointFileName)

        # Copy the template input file to the run directory
        copyfile(template_path, self.path)
        # Copy additional files
        for f in self.files2copy:
            copyfrom = os.path.join(os.getcwd(), f)
            copyto = os.path.join(directory, f)
            copyfile(copyfrom, copyto)
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
                sys.stderr.write(str((type(value))))
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
        # f = open(stdOut)
        # value = float(f.readline())
        # return LSValue(value)
        import pickle

        # os.path.join(directory, 'alematvol.xy')
        print(directory)
        sys.stderr.write(os.getcwd())

        os.system('cd ' + directory)
        sys.stderr.write(os.getcwd())

        # if os.path.exists('alematvol.xy'):
        if os.path.exists(os.path.join(directory, 'alematvol.xy')):

            settings_file_path = os.path.join(directory, 'settings.pkl')
            with open(settings_file_path, 'rb') as f:
                settings_dict = pickle.load(f)

            temp_file_path = os.path.join(directory, 'TempFilesPorosity')
            results_file_path = os.path.join(directory, 'ResultsFiles')

            print('Path for Temporary Files\n')
            print(temp_file_path)
            print('Path for Results:\n')
            print(results_file_path)

            # Create Directories for the result files
            if not os.path.isdir(temp_file_path):
                os.system('mkdir ' + temp_file_path)
            if not os.path.isdir(results_file_path):
                os.system('mkdir ' + results_file_path)

            print('Successfully created directories for result files.')

            # Post-processing: read some files created by LSDYNA, extract relevant information and save results in a pkl
            # file. Transform binout file into txt files that I can read and postprocess
            # os.system('l2a binout*')
            results_dict = {'sample': settings_dict['samples'][argument.index]}  # dictionary that will contain output results

            # read alematvol.xy file
            t, volume_solid, volume_void = read_alematvol_file(os.path.join(directory, 'alematvol.xy'))
            print('Successfully read alematvol.xy.')
            results_dict['alematvol.xy'] = {'t': t, 'volume_solid': volume_solid, 'volume_void': volume_void}

            # read elout file
            # Element stresses
            if os.path.isfile(os.path.join(directory, 'elout')):
                try:
                    t, av_stresses = read_elout_file(file_name=os.path.join(directory, 'elout'),
                                                     scale_to_MPa=settings_dict['scale_to_MPa'],
                                                     average_over_solid=True)
                    results_dict['elout'] = {'t': t, 'av_stresses_over_solid': av_stresses}
                    print('Successfully read elout.')
                except:
                    pass

            # read bndout file
            # Boundary Forces
            if os.path.isfile(os.path.join(directory, 'bndout')):
                try:
                    t, forces = read_bndout_file(file_name=os.path.join(directory, 'bndout'),
                                                 scale_to_N=settings_dict['scale_to_N'])
                    results_dict['bndout'] = {'t': t, 'forces': forces}
                    print('Successfully read bndout.')
                except:
                    pass

            # read dbfsi file
            # FSI Forces
            if os.path.isfile(os.path.join(directory, 'dbfsi')):
                try:
                    t, pressure, forces = read_dbfsi_file(file_name=os.path.join(directory, 'dbfsi'),
                                                          scale_to_MPa=settings_dict['scale_to_MPa'],
                                                          scale_to_N=settings_dict['scale_to_N'])
                    results_dict['dbfsi'] = {'t': t, 'pressure': pressure, 'forces': forces}
                    print('Successfully read dbfsi.')
                except:
                    pass

            # read spcforc file
            # Reaction forces at symmetry planes
            if os.path.isfile(os.path.join(directory, 'spcforc')):
                try:
                    t, forces = read_spcforc_file(file_name=os.path.join(directory, 'spcforc'),
                                                  scale_to_N=settings_dict['scale_to_N'])
                    results_dict['spcforc'] = {'t': t, 'forces': forces}
                    print('Successfully read spcforc.')
                except:
                    pass

            # dx_stresses = lcid_1(tvec=t_stresses, t1=1e-4, V=settings_dict['V_x'])
            print('Simulation Index:\n')
            print(argument.index)
            with open(results_file_path + '/results3_run{}.pkl'.format(argument.index), 'wb') as f:
                pickle.dump(results_dict, f)

            # Move some files to be kept somewhere safe :) then delete unnecessary stuff
            os.system('mv alematvol.xy ' + os.path.join(temp_file_path, 'alematvol_run{}.xy'.format(argument.index)))
            # os.system('mv elout ' + os.path.join(temp_file_path2, 'elout_run{}'.format(argument.index)))
            # command_delete = 'rm d3dump* d3full* adptmp *.inc *.tmp scr* disk* mes* kill* bg* load_profile* alematE*.xy ' \
            #                  'alematmas.xy binout*'
            # os.system(command_delete)

            value = True
            return LSValue(value)

        else:
            print('File does not exist')
            print(os.path.join(directory, 'alematvol.xy'))
            sys.stderr.write('File does not exist')
            sys.stderr.write(os.path.join(directory, 'alematvol.xy'))
            value = False
            return LSValue(value)


class LSValue(hms.Value, PickleableSWIG):

    def __init__(self, value):
        hms.Value.__init__(self)
        self.value = value
        return
