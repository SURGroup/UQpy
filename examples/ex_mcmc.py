from SampleMethods import *
from RunModel import RunModel
from module_ import def_model, def_target, readfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = 'input_mcmc.txt'
data = readfile(filename)

# extract input data
model = def_model(data['Model'])
method = data['Method']
N = data['Number of Samples']
d = data['Stochastic dimension']
pdf = data['Probability distribution (pdf)']
pdf_params = data['Probability distribution parameters']
pdf_proposal = data['Proposal distribution']
proposal_pdf_params = np.array(data['Proposal distribution parameters'])
pdf_target = def_target(data['Target distribution'])
marg_pdf_params = np.array(data['Marginal target distribution parameters'])
jump = data['Burn-in samples']
algorithm = data['MCMC algorithm']



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

'''
Run code
'''

mcmc = sm.MCMC(sm, N, target=pdf_target, MCMC_algorithm=algorithm, proposal=pdf_proposal, params=proposal_pdf_params,
               marginal_parameters=marg_pdf_params, njump=jump)
fx = rm.Evaluate(rm, mcmc.xi)


'''
Plots
'''


subpath = os.path.join(os.sep, path, 'mcmc')
os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)

np.savetxt('samples.txt', mcmc.xi, delimiter=' ')
np.savetxt('model.txt', fx.v)

plt.figure()
plt.scatter(mcmc.xi[:, 0], mcmc.xi[:, 1])
plt.savefig('samples.png')

plt.figure()
n, bins, patches = plt.hist(fx.v, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Histogram')
plt.savefig('histogram.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mcmc.xi[:, 0], mcmc.xi[:, 1], fx.v, c='r', s=2)
plt.gca().invert_xaxis()
plt.savefig('model.png')


os.chdir(current_dir)

