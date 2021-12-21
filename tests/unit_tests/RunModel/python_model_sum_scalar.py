import fire
import os


def python(index):
    command1 = "cp ./InputFiles/sum_scalar_" + str(index) + ".py ."
    # The user will need to modify command2 to point to the Matlab application on their system.
    command2 = "python3 sum_scalar_" + str(index) + ".py"
    command3 = "mv ./OutputFiles/oupt.npy ./OutputFiles/oupt_" + str(index) + ".npy"
    command4 = "rm sum_scalar_" + str(index) + ".py"
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)


if __name__ == '__main__':
    fire.Fire(python)
