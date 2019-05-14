import numpy as np
import os


def return_output(index):
    fnm = os.path.join(os.getcwd(), 'Output', 'output_element_{0}.csv'.format(index))
    with open(fnm, 'rb') as f:
        o = np.genfromtxt(f)

    return o
