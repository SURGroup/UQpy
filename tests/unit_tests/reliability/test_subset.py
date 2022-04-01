import shutil

import numpy as np

from UQpy.run_model.RunModel import RunModel
from UQpy.distributions.collection.Normal import Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sampling.MonteCarloSampling import MonteCarloSampling
from UQpy.reliability.SubsetSimulation import SubsetSimulation


def test_subset():  # Define the structural problem
    n_variables = 2
    model = 'pfn.py'
    Example = 'Example1'

    omega = 6
    epsilon = 0.01
    mu_m = 5
    sigma_m = 1
    mu_k = 125
    sigma_k = 25
    m = np.linspace(mu_m - 3 * sigma_m, mu_m + 3 * sigma_m, 101)
    d_m = Normal(loc=mu_m, scale=sigma_m)
    d_k = Normal(loc=mu_k, scale=sigma_k)
    dist_nominal = JointIndependent(marginals=[d_m, d_k])

    from UQpy.sampling import Stretch

    n_samples_set = 1000
    p_cond = 0.1
    n_chains = int(n_samples_set * p_cond)

    mc = MonteCarloSampling(distributions=dist_nominal,
                            nsamples=n_samples_set,
                            random_state=1)

    init_sus_samples = mc.samples
    RunModelObject_SuS = RunModel(model_script=model, model_object_name=Example)

    sampling = Stretch(pdf_target=dist_nominal.pdf, dimension=2, n_chains=1000, random_state=0)

    SuS_object = SubsetSimulation(sampling=sampling, runmodel_object=RunModelObject_SuS, conditional_probability=p_cond,
                                nsamples_per_subset=n_samples_set, samples_init=init_sus_samples)

    print(SuS_object.failure_probability)
    assert SuS_object.failure_probability == 3.1200000000000006e-05
    shutil.rmtree(RunModelObject_SuS.model_dir)






