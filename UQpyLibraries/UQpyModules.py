import numpy as np


class RunCommandLine:

    # python UQpy_.py
    def __init__(self, argparseobj):
        # Defaults
        self.args = argparseobj

        # Actually Run UQpy
        self.run_uq()

    def run_uq(self):

        print("\nExecuting UQpy from commandline:\n")

        # Read  Input file
        from UQpyLibraries import ReadInputFile

        data = ReadInputFile.readfile(self.args.InputFile, self.args.Working_directory)
        init_sm(data)   # initialize the sampling method
        samples = run_sm(data)    # run the sampling method
        save_txt(samples, 'samples.txt', self.args.Output_directory)  # save the samples in a *.txt
        if self.args.ModelFile is not None:
            model = run_model(self.args.ModelFile, self.args.Working_directory, self.args.Output_directory)
            save_txt(model, 'model.txt', self.args.Output_directory)  # save the model evaluations in a *.txt

        print("\nSuccessful execution of UQpy\n\n")


def init_sm(data):

    if 'Method' in data.keys():
        if data['Method'] not in ['mcs', 'lhs', 'mcmc', 'pss', 'sts', 'SuS']:
            raise NotImplementedError("Method - %s not available" % data['Method'])
        else:
            print("\nInitializing method: %k \n", data['Method'])
    else:
        raise NotImplementedError("No sampling method was provided")

    if 'Number of Samples' not in data.keys():
        data['Number of Samples'] = 10
        raise NotImplementedError("Number of samples not provided- Set default : %s " % 10)

    if data['Method'] == 'mcs':
        if 'Probability distribution (pdf)' not in data.keys():
            raise NotImplementedError("Probability distribution not provided")
        elif 'Probability distribution parameters' not in data.keys():
            raise NotImplementedError("Probability distribution parameters not provided")

    if data['Method'] == 'lhs':
        if 'Probability distribution (pdf)' not in data.keys():
            raise NotImplementedError("Probability distribution not provided")
        if 'Probability distribution parameters' not in data.keys():
            raise NotImplementedError("Probability distribution parameters not provided")
        if 'LHS criterion' not in data.keys():
            data['LHS criterion'] = 'centered'
            raise Warning("LHS criterion not defined. The default is centered")
        if 'distance metric' not in data.keys():
            data['distance metric'] = 'euclidean'
            raise Warning("Distance metric for the LHS not defined. The default is Euclidean")
        if 'iterations' not in data.keys():
            data['iterations'] = 1000
            raise Warning("Iterations for the LHS not defined. The default number is 1000")

    elif data['Method'] == 'mcmc':
        if 'MCMC algorithm' not in data.keys():
            raise NotImplementedError("MCMC algorithm not provided")
        if 'Proposal distribution' not in data.keys():
            raise NotImplementedError("Proposal distribution not provided")
        if 'Proposal distribution parameters' not in data.keys():
            raise NotImplementedError("Proposal distribution parameters (width) not provided")
        if 'Target distribution' not in data.keys():
            raise NotImplementedError("Target distribution not provided")
        if 'Marginal target distribution parameters' not in data.keys():
            raise NotImplementedError("Target distribution parameters not provided")
        if 'Burn-in samples' not in data.keys():
            data['Burn-in samples'] = 1
            raise Warning("Number of samples to skip in order to avoid Burn-in not provided."
                          "The default will be set equal to 1")


def run_sm(data):
    if data['Method'] == 'mcs':
        from UQpyLibraries.SampleMethods import MCS
        print("\nRunning  %k \n", data['Method'])
        x = MCS(pdf=data['Probability distribution (pdf)'],
                           pdf_params=data['Probability distribution parameters'],
                           nsamples=data['Number of Samples'])

    elif data['Method'] == 'lhs':
        from UQpyLibraries.SampleMethods import LHS
        print("\nRunning  %k \n", data['Method'])
        x = LHS(pdf=data['Probability distribution (pdf)'],
                           pdf_params=data['Probability distribution parameters'],
                           nsamples=data['Number of Samples'], lhs_criterion=data['LHS criterion'],
                           lhs_metric=data['distance metric'], lhs_iter=data['iterations'])
    samples = x.samples
    print(samples)
    return samples


def run_model(script, working_dir, output_dir):
    import os
    current_dir = os.getcwd()
    file_path = os.path.join(os.sep, current_dir, working_dir)
    os.chdir(file_path)
    script = './bash_test.sh'
    print("\nEvaluating the model:\n")
    os.system(script)  # run model
    os.chdir(current_dir)
    # Load the model evaluations from the .txt
    file_path = os.path.join(os.sep, current_dir, output_dir)
    os.chdir(file_path)
    model = load_txt('model.txt')    # load the saved model
    print(model)

    return model


def save_txt(input, filename, output_dir):
    import os
    current_dir = os.getcwd()
    file_path = os.path.join(os.sep, current_dir, output_dir)
    os.chdir(file_path)
    np.savetxt(filename, input)
    os.chdir(current_dir)


def load_txt(filename):
    return np.loadtxt(filename)
