import numpy as np
import os

x = np.zeros(3)
x[0] = <x0>
x[1] = <x1>
x[2] = <x2>
output = sum(x)

if not os.path.isdir('OutputFiles'):
    os.mkdir('OutputFiles')

np.save('OutputFiles/oupt.npy', output)
