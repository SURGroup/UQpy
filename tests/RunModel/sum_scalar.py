import numpy as np
import os

x = np.zeros(3)
x[0] = <var1>
x[1] = <var11>
x[2] = <var111>
output = sum(x)

if not os.path.isdir('OutputFiles'):
    os.mkdir('OutputFiles')

np.save('OutputFiles/oupt.npy', output)
