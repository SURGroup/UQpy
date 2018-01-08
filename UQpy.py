from SampleMethods import *
from RunModel import RunModel
from module_ import handle_input_file, def_model, def_target
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


filename = 'input_srom.txt'

current_dir = os.getcwd()

path = os.path.join(os.sep, current_dir, 'examples')
os.makedirs(path, exist_ok=True)
os.chdir(path)


if filename == 'input_mcmc.txt':
    _model, method, nsamples, dimension, distribution, parameters, x0, MCMC_algorithm, params,proposal, target, jump = handle_input_file(filename)
    target = def_target(target)

elif filename == 'input_lhs.txt':
    _model, method, nsamples, dimension, distribution, parameters, lhs_criterion, dist_metric,iterations = handle_input_file(filename)

elif filename == 'input_mcs.txt':
    _model, method, nsamples, dimension, distribution, parameters = handle_input_file(filename)

elif filename == 'input_pss.txt':
    _model, method, nsamples, dimension, distribution, parameters, pss_design, pss_stratum=handle_input_file(filename)

elif filename == 'input_sts.txt':
    _model, method, nsamples, dimension, distribution, parameters, sts_input = handle_input_file(filename)

elif filename == 'input_srom.txt':
    _model, method, nsamples, dimension, distribution, sts_input, moments, parameters, properties, weights_errors, \
    weights_samples = handle_input_file(filename)

os.chdir(current_dir)

model = def_model(_model)
sm = SampleMethods(dimension=dimension, distribution=distribution, parameters=parameters, method=method)

path = os.path.join(os.sep, current_dir, 'results')
os.makedirs(path, exist_ok=True)
os.chdir(path)

if method == 'mcs':
    g = RunModel(generator=sm,   nsamples=nsamples,  method=method,  model=model)
    subpath = os.path.join(os.sep, path, 'mcs')

elif method == 'lhs':
    g = RunModel(generator=sm,  nsamples=nsamples,  method=method,  model=model, lhs_criterion='random')
    subpath = os.path.join(os.sep, path, 'lhs')

elif method == 'mcmc':
    g = RunModel(generator=sm, nsamples=nsamples, method=method, model=model,  x0=x0, MCMC_algorithm=MCMC_algorithm, proposal=proposal, params=params, target=target, jump=jump)
    subpath = os.path.join(os.sep, path, 'mcmc')

elif method == 'pss':
    g = RunModel(generator=sm,  method=method, model=model,   pss_design=pss_design, pss_stratum=pss_stratum)
    subpath = os.path.join(os.sep, path, 'pss')

elif method == 'sts':
    g = RunModel(generator=sm,  method=method, model=model, sts_input=sts_input)
    subpath = os.path.join(os.sep, path, 'sts')

elif method == 'srom':
    g = RunModel(generator=sm, method=method, model=model, sts_input=sts_input, moments=moments, properties=properties, weights_errors=weights_errors, weights_samples=weights_samples)
    subpath = os.path.join(os.sep, path, 'srom')

os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)


if method == 'srom':
    g.eval = np.reshape(g.eval, (27, 4))
    g.eval = g.eval[np.argsort(g.eval[:, 0].flatten())]
    g.eval = np.reshape(g.eval, (27, 4))
    plt.plot(g.eval[:, 0], np.reshape(np.cumsum(g.eval[:, 3]), [27, 1]), label='Approximate')
    counts, bin_edges = np.histogram(g.mcs[:, 0], bins=100, normed=True)
    cdf = np.cumsum(counts)
    cdf = (1 / cdf[np.size(counts) - 1]) * cdf
    plt.plot(bin_edges[1:], cdf, label='Exact')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig('eigen.png')
else:
    np.savetxt('samples.txt', g.samples, delimiter=' ')
    np.savetxt('model.txt', g.eval)

    plt.figure()
    plt.scatter(g.samples[:, 0], g.samples[:, 1])
    plt.savefig('samples.png')

    plt.figure()
    n, bins, patches = plt.hist(g.eval[:, 0], 50, normed=1, facecolor='g', alpha=0.75)
    plt.title('Histogram')
    plt.savefig('histogram.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(g.samples[:, 0], g.samples[:, 1], g.samples[:, 2], c='r', s=2)
    plt.gca().invert_xaxis()
    plt.savefig('model.png')


os.chdir(current_dir)



