import numpy as np


class ReadOutput:
    def __init__(self, index):
        self.qoi = np.load("./OutputFiles/oupt_%d.npy" % index)
