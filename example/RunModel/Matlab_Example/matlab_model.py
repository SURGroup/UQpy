import fire
import os


def matlab(index):
    command1 = "cp ./InputFiles/dummy_model_" + str(index) + ".number_of_variables ."
    command2 = "/Applications/MATLAB_R2018a.app/bin/matlab " \
               "-nosplash -nojvm -nodisplay -nodesktop -r 'run dummy_model_" + str(index) + ".number_of_variables; exit'"
    command3 = "mv ./OutputFiles/oupt.out ./OutputFiles/oupt_" + str(index) + ".out"
    command4 = "rm dummy_model_" + str(index) + ".number_of_variables"
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)


if __name__ == '__main__':
    fire.Fire(matlab)
