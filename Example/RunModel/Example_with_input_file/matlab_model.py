import fire
import os


def matlab(index):
    command = "matlab -nosplash -nojvm -nodisplay -nodesktop -r 'run dummy_model(" + str(index + 1) + ");exit'"
    os.system(command)


if __name__ == '__main__':
    fire.Fire(matlab)
