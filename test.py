from functools import partial
from SampleMethods import *
from RunModel import RunModel
import itertools

'''
dimension = int(sys.argv[1])
distribution_type = sys.argv[2]
method = sys.argv[3]
nsamples = int(sys.argv[4])
'''

dimension = 2
distribution_type = 'Uniform'
method = 'sts'
nsamples = 100
_model = model_zabaras

print(_model)
distribution = list(itertools.repeat(distribution_type, dimension))
parameters = np.identity(dimension)
model = partial(_model)
target = partial(marginal)
sm = SampleMethods(dimension=dimension, distribution=distribution, parameters=parameters)

print()

if method == 'mcs':
    g = RunModel(generator=sm,   nsamples=nsamples,  method=method,  model=model)
elif method == 'lhs':
    g = RunModel(generator=sm,  nsamples=nsamples,  method=method,  model=model, lhs_criterion='random')
elif method == 'mcmc':
    g = RunModel(generator=sm, nsamples=nsamples,  method=method, model=model,  MCMC_algorithm='MMH', target=target)
elif method == 'pss':
    g = RunModel(generator=sm,  method=method, model=model,   pss_design=[2, 2], pss_stratum=[25, 25])
elif method == 'sts':
    g = RunModel(generator=sm,  method=method, model = model, sts_input=[10, 10])


print('End with success')



