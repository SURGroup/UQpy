from SampleMethods import *
from RunModel import RunModel
from module_ import handle_input_file, def_model, def_target
from Reliability import ReliabilityMethods
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


filename = 'input_SuS.txt' #sys.argv[1]

current_dir = os.getcwd()

path = os.path.join(os.sep, current_dir, 'examples')
os.makedirs(path, exist_ok=True)
os.chdir(path)


if filename == 'input_mcmc.txt':
    _model, method, nsamples, dimension, distribution, parameters, x0, MCMC_algorithm, params,proposal, target, \
    jump = handle_input_file(filename)
    target = def_target(target)

elif filename == 'input_lhs.txt':
    _model, method, nsamples, dimension, distribution, parameters, lhs_criterion, dist_metric,\
    iterations = handle_input_file(filename)

elif filename == 'input_mcs.txt':
    _model, method, nsamples, dimension, distribution, parameters = handle_input_file(filename)

elif filename == 'input_pss.txt':
    _model, method, nsamples, dimension, distribution, parameters, pss_design, pss_stratum=handle_input_file(filename)

elif filename == 'input_sts.txt':
    _model, method, nsamples, dimension, distribution, parameters, sts_input = handle_input_file(filename)

elif filename == 'input_SuS.txt':
    _model, method, nsamples_per_subset, dimension, distribution,  MCMC_algorithm, params,proposal, proposal_width, \
    target, conditional_prob, Yf, marginal_param = handle_input_file(filename)



os.chdir(current_dir)

model = def_model(_model)

path = os.path.join(os.sep, current_dir, 'results')
os.makedirs(path, exist_ok=True)
os.chdir(path)

if method == 'mcs':
    g = RunModel(nsamples=nsamples, dimension=dimension, method=method,  model=model)
    subpath = os.path.join(os.sep, path, 'mcs')

elif method == 'lhs':
    g = RunModel(nsamples=nsamples, dimension=dimension, method=method,  model=model, lhs_criterion='random')
    subpath = os.path.join(os.sep, path, 'lhs')

elif method == 'mcmc':
    target = def_target(target)
    g = RunModel(nsamples=nsamples, dimension=dimension, method=method, model=model,  x0=x0,
                 MCMC_algorithm=MCMC_algorithm, proposal=proposal, params=params, target=target, jump=jump)
    subpath = os.path.join(os.sep, path, 'mcmc')

elif method == 'pss':
    g = RunModel(method=method, dimension=dimension, model=model,   pss_design=pss_design, pss_stratum=pss_stratum)
    subpath = os.path.join(os.sep, path, 'pss')

elif method == 'sts':
    g = RunModel(method=method, dimension=dimension, model = model, sts_input=sts_input)
    subpath = os.path.join(os.sep, path, 'sts')

elif method == 'SuS':
    target = def_target(target)
    g = ReliabilityMethods.SubsetSimulation(dimension=dimension, nsamples_per_subset=nsamples_per_subset,  model=model,
                                            MCMC_algorithm=MCMC_algorithm, proposal=proposal,
                                            conditional_prob=conditional_prob, marginal_params=marginal_param,
                                            proposal_width=proposal_width, params=params, target=target, limit_state=Yf)
    subpath = os.path.join(os.sep, path, 'SuS')


os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)

np.savetxt('samples.txt', g.samples, delimiter=' ')
np.savetxt('model.txt', g.eval)

plt.figure()
plt.scatter(g.samples[:, 0], g.samples[:, 1])
plt.savefig('samples.png')

plt.figure()
n, bins, patches = plt.hist(g.eval, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Histogram')
plt.savefig('histogram.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(g.samples[:, 0], g.samples[:, 1], g.eval, c='r', s=2)
plt.gca().invert_xaxis()
plt.savefig('model.png')


os.chdir(current_dir)



