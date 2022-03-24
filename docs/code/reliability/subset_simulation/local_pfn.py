"""

Auxiliary File
======================================================================

"""
import numpy as np


def run_python_model(samples, b_eff, d):
    return [b_eff * np.sqrt(d) - np.sum(samples[i, :]) for i in range(samples.shape[0])]
