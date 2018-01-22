from UQpyLibraries.UQpyModules import *

mydict = {}
mydict['Method'] = 'mcs'
mydict['Probability distribution (pdf)'] = ['Uniform', 'Uniform']
mydict['Probability distribution parameters'] = [[0, 1], [0, 1]]
mydict['Number of Samples'] = 10


init_sm(mydict)
samples = run_sm(mydict)
model = run_model('./bash_test.sh', 'examples', 'simUQpyOut')
print()