import UQpy.RunModel
import os
import shutil
import numpy as np
import sys


class RunCommandLine:

    def __init__(self, argparseobj):
        os.system('clear')
        self.args = argparseobj

        ################################################################################################################
        # Read  UQpy parameter file

        os.chdir(os.path.join(os.getcwd(), self.args.Model_directory))
        if not os.path.isfile('UQpy_Params.txt'):
            print("Error: UQpy parameters file does not exist")
            sys.exit()
        else:
            from UQpy.ReadInputFile import readfile
            data = readfile('UQpy_Params.txt')

        ################################################################################################################
        # Run UQpy

        print("\nExecuting UQpy...\n")
        ################################################################################################################

        # Run Selected method
        if data['method'] in ['SuS']:
            self.run_reliability(data)
        elif data['method'] in ['mcs', 'lhs', 'mcmc', 'pss', 'sts']:
            self.run_uq(data)

    def run_uq(self, data):

        # Steps:
        # Initialize the sampling method (Check if UQpy_Params.txt contains all the necessary information)
        # Actually run the selected sampling method in U(0, 1) and transform to the original space
        # Save the samples in a .txt (or .csv) file
        # If a solver (black box model) is provided then:
        #                              If parallel processing is selected: Split the samples into chunks
        # Run the model
        # Save the model evaluations

        # Brute-force sampling methods. Set the adaptivity flag False
        self.args.Adaptive = False
        ################################################################################################################
        # Initialize the requested UQpy method: Check if all necessary parameters are defined in the UQpyParams.txt file
        from UQpy.SampleMethods import init_sm, run_sm
        init_sm(data)

        ################################################################################################################
        # Run the requested UQpy method
        rvs = run_sm(data)

        # Save the samples in a .txt file
        np.savetxt('UQpy_Samples.txt', rvs.samples, fmt='%0.5f')

        # Save the samples in a .csv file
        if 'names of parameters' not in data:
            import itertools
            data['names of parameters'] = list(itertools.repeat('#name', rvs.samples.shape[1]))

        save_csv(data['names of parameters'], rvs.samples)

        ################################################################################################################
        # If a model is provided then run it

        if self.args.Solver is not None:
            UQpy.RunModel(cpu=self.args.CPUs, model_script=self.args.Solver, input_script=self.args.Input_Shell_Script,
                          output_script=self.args.Output_Shell_Script, dimension=rvs.dimension)



        ################################################################################################################
        print("\nSuccessful execution of UQpy\n\n")

    def run_reliability(self, data):
        from UQpy.Reliability import init_rm, run_rm
        init_rm(data)
        if data['method'] == 'SuS':
            from UQpy.Reliability import SubsetSimulation
            self.args.CPUs_flag = True
            self.args.ParallelProcessing = False
            self.args.Adaptive = True
            sus = run_rm(self, data)

            # Save the samples in a .txt file
            np.savetxt('UQpy_Samples.txt', sus.samples, fmt='%0.5f')

            # Save the samples in a .csv file
            save_csv(data['Names of random variables'], sus.samples)

            # Save the probability of failure in a .txt file
            print(sus.pf)
            with open('PF.txt', 'wb') as f:
                np.savetxt(f, [sus.pf], fmt='%0.6f')
                np.savetxt(f, [sus.cov], fmt='%0.6f')
        ################################################################################################################
        # Move the data to directory simUQpyOut/ , delete the temp/ directory
        # and terminate the program

        _files = list()
        _files.append('UQpy_Samples.csv')
        _files.append('UQpy_Samples.txt')
        _files.append('PF.txt')

        for file_name in _files:
            full_file_name = os.path.join(self.args.WorkingDir, file_name)
            shutil.copy(full_file_name, self.args.Output_directory)

        shutil.rmtree(self.args.WorkingDir)
        shutil.move(self.args.Output_directory, self.args.Model_directory)

        ################################################################################################################
        print("\nSuccessful execution of UQpy\n\n")


def save_csv(headers, param_values):

    index = np.array(range(1, param_values.shape[0] + 1)).astype(int)
    param_values = np.hstack((index.reshape(index.shape[0], 1), param_values))
    expand_header = list()
    expand_header.append('Run')
    for i in range(len(headers)):
        expand_header.append(headers[i])
    import csv
    with open('UQpy_Samples.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(expand_header)
        for val in param_values:
            writer.writerow(val)


def save_txt(headers, param_values):
    index = np.array(range(1, param_values.shape[0] + 1)).astype(int)
    param_values = np.hstack((index.reshape(index.shape[0], 1), param_values))
    expand_header = list()
    expand_header.append('Run')
    for i in range(len(headers)):
        expand_header.append(headers[i])

    header = ', '.join(expand_header)
    np.savetxt('UQpy_Samples.txt', param_values, header=str(header), fmt='%0.5f')

