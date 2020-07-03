import fire
import os
import numpy as np
import subprocess


def extract_disp_temp_output(index):
    # index = int(index)
    output_script_path = os.path.join(os.getcwd(), 'abaqus_output_script.py')
    command = "abaqus cae nogui=" + output_script_path
    try:
        out = os.system(command)
        print('Out: ', out)
        if out == 0:
            print('Example: Successful output extraction.')
            outfilename = 'time_temp_disp_data.csv'
            data = np.genfromtxt(outfilename, delimiter=',')

            # Delete the odb file after extracting output
            dir_name = os.getcwd()
            test = os.listdir(dir_name)
            for item in test:
                if item.endswith(".odb"):
                    os.remove(os.path.join(dir_name, item))

            # Compute the maximum allowable displacement
            length_of_beam = 1  # in number_of_variables
            depth_of_beam = 0.035  # in number_of_variables
            max_allowable_disp = length_of_beam ** 2 / (400 * depth_of_beam)

            # Obtain the maximum midpoint displacement
            midpoint_disps = data[:, 2]
            max_midpoint_disps = max(map(abs, midpoint_disps))

            # Performance function
            Y = max_allowable_disp - max_midpoint_disps
            return Y

    except OSError as err:
        print(err)
        return np.array([100, 100, 10000])


if __name__ == '__main__':
    fire.Fire(extract_disp_temp_output)
