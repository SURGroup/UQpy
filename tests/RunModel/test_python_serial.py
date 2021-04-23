from UQpy.SampleMethods import MCS
from UQpy.RunModel import RunModel
from UQpy.Distributions import Normal
import numpy as np
import sys
import os

d = Normal(loc=0, scale=1)
x_mcs = MCS(dist_object=[d, d, d], nsamples=5, random_state=1234)

m11 = RunModel(ntasks=1, model_script='/tests/RunModel/python_model.py', model_object_name='SumRVs', model_dir='temp', verbose=True)
m11.run(samples=x_mcs.samples, )



def test_python_serial():
    print(os.getcwd())
    m11 = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', model_dir='temp', verbose=True)
    m11.run(samples=x_mcs.samples, )
    assert m11.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787, -2.3182103441722544, -3.297351053358514]


def test_python_parallel():
    m11 = RunModel(ntasks=3, model_script='python_model.py', model_object_name='SumRVs', model_dir='temp', verbose=True)
    m11.run(samples=x_mcs.samples, )
    assert m11.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787, -2.3182103441722544, -3.297351053358514]
