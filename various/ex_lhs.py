from UQpyLibraries.SampleMethods import *
from various.RunModel import RunModel
from various.module_ import def_model, readfile
import matplotlib.pyplot as plt

filename = 'input_lhs.txt'
data = readfile(filename)

# extract input data
model = def_model(data['Model'])
method = data['Method']
N = data['Number of Samples']
d = data['Stochastic dimension']
pdf = data['Probability distribution (pdf)']
pdf_params = data['Probability distribution parameters']
LHS_criterion = data['LHS criterion']
dist_metric = data['distance metric']
iterations = data['iterations']

current_dir = os.getcwd()
path = os.path.join(os.sep, current_dir, 'results')
os.makedirs(path, exist_ok=True)
os.chdir(path)


'''
Initialize 
1. class Sampling methods
2. class RunModel
'''

sm = SampleMethods(distribution=pdf, dimension=d, parameters=pdf_params, method=method)
rm = RunModel(model=model)

lhs = sm.LHS(sm, d, N, LHS_criterion, iterations, dist_metric)
fx = rm.Evaluate(rm, lhs.xi)


subpath = os.path.join(os.sep, path, 'lhs')
os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)

np.savetxt('samples.txt', lhs.xi, delimiter=' ')
np.savetxt('model.txt', fx.v)

plt.figure()
plt.scatter(lhs.xi[:, 0], lhs.xi[:, 1])
plt.savefig('samples.png')

plt.figure()
n, bins, patches = plt.hist(fx.v, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Histogram')
plt.savefig('histogram.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lhs.xi[:, 0], lhs.xi[:, 1], fx.v, c='r', s=2)
plt.gca().invert_xaxis()
plt.savefig('model.png')


os.chdir(current_dir)