import os
import fire


def run_single_element(index):
    index = int(index)

    input_file_path = os.path.join(os.getcwd(), 'InputFiles', 'single_element_' + str(index) + ".inp")

    command1 = ("abaqus job=single_element_simulation_" + str(index) +
                " input=" + input_file_path + " cpus=1 interactive")

    command2 = ("abaqus python process_odb.py " + str(index))

    os.system(command1)
    os.system(command2)


if __name__ == '__main__':
    fire.Fire(run_single_element)
