import numpy as np


def model(inputs=None):
    if inputs is not None:
        
        x = inputs[0][0]
        y = inputs[0][1]
        inside = _check_if_inside(x, y)

    return inside

def _check_if_inside(x, y):
    if x ** 2 + y ** 2 < 1:
        inside = 1
    else:
        inside = 0
    return inside

