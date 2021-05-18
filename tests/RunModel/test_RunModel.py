from UQpy.SampleMethods import MCS
from UQpy.RunModel import RunModel
from UQpy.Distributions import Normal
import pytest
import shutil
import os
from pathlib import Path
import numpy as np


d = Normal(loc=0, scale=1)
x_mcs = MCS(dist_object=[d, d, d], nsamples=5, random_state=1234)


def test_div_zero():
    print(os.getcwd())
    with pytest.raises(TypeError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', fmt=20, delete_files=True)


def test_fmt():
    with pytest.raises(TypeError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', fmt=20, delete_files=True)


def test_var_names():
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', var_names=[20],
                 delete_files=True)


def test_model_script():
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='random_file_name', model_object_name='SumRVs', delete_files=True)


def test_samples():
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', samples="samples_string",
                 delete_files=True)


def test_python_serial_workflow_class_vectorized():
    model_python_serial_class = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', verbose=True)
    model_python_serial_class.run(samples=x_mcs.samples)
    assert model_python_serial_class.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787,
                                                  -2.3182103441722544, -3.297351053358514]


def test_python_serial_workflow_class():
    model_python_serial_class = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', vec=False, verbose=True)
    model_python_serial_class.run(samples=x_mcs.samples)
    assert model_python_serial_class.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787,
                                                  -2.3182103441722544, -3.297351053358514]


def test_python_serial_workflow_function_vectorized():
    model_python_serial_function = RunModel(ntasks=1, model_script='python_model.py', model_object_name='sum_rvs', verbose=True)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert model_python_serial_function.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787,
                                                     -2.3182103441722544, -3.297351053358514]


def test_python_serial_workflow_function():
    model_python_serial_function = RunModel(ntasks=1, model_script='python_model.py', model_object_name='sum_rvs', vec=False, verbose=True)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert model_python_serial_function.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787,
                                                     -2.3182103441722544, -3.297351053358514]


def test_python_parallel_workflow_class():
    model_python_parallel_class = RunModel(ntasks=3, model_script='python_model.py', model_object_name='SumRVs', verbose=True)
    model_python_parallel_class.run(samples=x_mcs.samples)
    assert model_python_parallel_class.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787,
                                                    -2.3182103441722544, -3.297351053358514]


def test_python_parallel_workflow_function():
    model_python_parallel_function = RunModel(ntasks=3, model_script='python_model.py', model_object_name='sum_rvs', verbose=True)
    model_python_parallel_function.run(samples=x_mcs.samples)
    assert model_python_parallel_function.qoi_list == [2.5086338287600496, 0.6605587413536298, 1.7495075921211787,
                                                       -2.3182103441722544, -3.297351053358514]


def test_third_party_serial():
    names = ['var1', 'var11', 'var111']
    m = RunModel(ntasks=1, model_script='python_model_sum_scalar.py',
                 input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, fmt="{:>10.4f}", verbose=True, delete_files=True)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list), np.array([2.5086, 0.6605, 1.7495, -2.3183, -3.2974]))


def test_third_party_parallel():
    names = ['var1', 'var11', 'var111']
    m = RunModel(ntasks=3, model_script='python_model_sum_scalar.py',
                 input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, fmt="{:>10.4f}", verbose=True, delete_files=True)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list), np.array([2.5086, 0.6605, 1.7495, -2.3183, -3.2974]))
