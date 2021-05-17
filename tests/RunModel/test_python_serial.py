from UQpy.SampleMethods import MCS
from UQpy.RunModel import RunModel
from UQpy.Distributions import Normal
import pytest
import shutil
import os

# sys.path.append()

d = Normal(loc=0, scale=1)
x_mcs = MCS(dist_object=[d, d, d], nsamples=5, random_state=1234)
# copy the model file to the parent directory
# shutil.copy2('python_model.py', )


def test_div_zero():
    with pytest.raises(TypeError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', fmt=20, delete_files=True)


def test_fmt():
    with pytest.raises(TypeError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', fmt=20, delete_files=True)


def test_var_names():
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', var_names=[20], delete_files=True)


def test_model_script():
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='random_file_name', model_object_name='SumRVs', delete_files=True)


def test_samples():
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', samples="samples_string", delete_files=True)


def test_serial_input():
    m11 = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', model_dir='temp',
                   verbose=True)
    m11.run(samples=x_mcs.samples, )
    assert (m11.qoi_list, [2.5086338287600496, 0.6605587413536298, 1.7495075921211787, -2.3182103441722544, -3.297351053358514])


def test_python_parallel():
    m11 = RunModel(ntasks=3, model_script='python_model.py', model_object_name='SumRVs', model_dir='temp',
                   verbose=True)
    m11.run(samples=x_mcs.samples, )
    assert(m11.qoi_list, [2.5086338287600496, 0.6605587413536298, 1.7495075921211787, -2.3182103441722544, -3.297351053358514])


# def test_third_party_serial():
#     names = ['var1', 'var11', 'var111']
#     m = RunModel(ntasks=1, model_script='matlab_model_sum_scalar.py',
#                  input_template='sum_scalar.m', var_names=names, model_object_name="matlab",
#                  output_script='process_matlab_output.py', output_object_name='read_output',
#                  resume=False, model_dir='Matlab_Model', fmt="{:>10.4f}", verbose=True)
#     m.run(x_mcs.samples)
#     print(m.qoi_list)
#     assert(1, 1)
