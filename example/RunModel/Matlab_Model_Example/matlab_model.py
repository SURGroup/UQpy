import fire
import os


def matlab(index):
    command1 = "cp ./InputFiles/dummy_model_" + str(index + 1) + ".m ."
    command2 = "matlab -nosplash -nojvm -nodisplay -nodesktop -r 'run dummy_model_" + str(index + 1) + ".m; exit'"
    command3 = "mv ./OutputFiles/oupt.out ./OutputFiles/oupt_" + str(index + 1) + ".out"
    command4 = "rm dummy_model_" + str(index + 1) + ".m"
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)


if __name__ == '__main__':
    fire.Fire(matlab)
