from UQpy.SampleMethods import MCS
from UQpy.RunModel import RunModel
from UQpy.Distributions import Normal
import pytest
import os
import numpy as np


d = Normal(loc=0, scale=1)
x_mcs = MCS(dist_object=[d, d, d], nsamples=5, random_state=1234)
x_mcs_new = MCS(dist_object=[d, d, d], nsamples=5, random_state=2345)
verbose_parameter = True
os.chdir('./tests/RunModel')


def test_div_zero():
    print(os.getcwd())
    with pytest.raises(TypeError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', fmt=20, delete_files=True)


def test_fmt_1():
    with pytest.raises(TypeError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', fmt=20, delete_files=True)


def test_fmt_2():
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs', fmt="random_string",
                 delete_files=True)


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
    model_python_serial_class = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs',
                                         verbose=verbose_parameter)
    model_python_serial_class.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_python_serial_workflow_class():
    model_python_serial_class = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs',
                                         vec=False, verbose=verbose_parameter)
    model_python_serial_class.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_direct_samples():
    model_python_serial_class = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs',
                                         vec=False, verbose=verbose_parameter, samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_append_samples_true():
    model_python_serial_class = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs',
                                         vec=False, verbose=verbose_parameter, samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))
    model_python_serial_class.run(x_mcs_new.samples, append_samples=True)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(),
                       np.sum(np.vstack((x_mcs.samples, x_mcs_new.samples)), axis=1))


def test_append_samples_false():
    model_python_serial_class = RunModel(ntasks=1, model_script='python_model.py', model_object_name='SumRVs',
                                         vec=False, verbose=verbose_parameter, samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))
    model_python_serial_class.run(x_mcs_new.samples, append_samples=False)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs_new.samples, axis=1))


def test_python_serial_workflow_function_vectorized():
    model_python_serial_function = RunModel(ntasks=1, model_script='python_model.py', model_object_name='sum_rvs',
                                            verbose=verbose_parameter)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_python_serial_workflow_function():
    model_python_serial_function = RunModel(ntasks=1, model_script='python_model.py', model_object_name='sum_rvs',
                                            vec=False, verbose=verbose_parameter)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_python_serial_workflow_function_no_object_name():
    model_python_serial_function = RunModel(ntasks=1, model_script='python_model_function.py', vec=False,
                                            verbose=verbose_parameter)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_python_serial_workflow_class_no_object_name():
    model_python_serial_function = RunModel(ntasks=1, model_script='python_model_class.py', vec=False,
                                            verbose=verbose_parameter)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_python_parallel_workflow_class():
    model_python_parallel_class = RunModel(ntasks=3, model_script='python_model.py', model_object_name='SumRVs',
                                           verbose=verbose_parameter)
    model_python_parallel_class.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_parallel_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_python_parallel_workflow_function():
    model_python_parallel_function = RunModel(ntasks=3, model_script='python_model.py', model_object_name='sum_rvs',
                                              verbose=verbose_parameter)
    model_python_parallel_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_parallel_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_third_party_serial():
    names = ['var1', 'var11', 'var111']
    m = RunModel(ntasks=1, model_script='python_model_sum_scalar.py',
                 input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
                 output_script='process_third_party_output.py', output_object_name='read_output',
                 resume=False, fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)


def test_third_party_serial_output_class():
    names = ['var1', 'var11', 'var111']
    m = RunModel(ntasks=1, model_script='python_model_sum_scalar.py',
                 input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
                 output_script='process_third_party_output_class.py', output_object_name='ReadOutput',
                 resume=False, fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)


def test_third_party_serial_no_output_class():
    names = ['var1', 'var11', 'var111']
    m = RunModel(ntasks=1, model_script='python_model_sum_scalar.py', input_template='sum_scalar.py', var_names=names,
                 model_object_name="matlab", output_script='process_third_party_output_class.py', resume=False,
                 fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)


def test_third_party_serial_no_output_function():
    names = ['var1', 'var11', 'var111']
    m = RunModel(ntasks=1, model_script='python_model_sum_scalar.py', input_template='sum_scalar.py', var_names=names,
                 model_object_name="matlab", output_script='process_third_party_output.py', resume=False,
                 fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)


def test_third_party_parallel():
    names = ['var1', 'var11', 'var111']
    m = RunModel(ntasks=3, model_script='python_model_sum_scalar.py',
                 input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
                 output_script='process_third_party_output.py', output_object_name='read_output',
                 resume=False, fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)


def test_third_party_default_var_names():
    model_third_party_default_names = RunModel(ntasks=1, model_script='python_model_sum_scalar_default.py',
                                               input_template='sum_scalar_default.py', model_object_name="python",
                                               output_script='process_third_party_output.py',
                                               output_object_name='read_output',
                                               resume=False, fmt="{:>10.4f}", verbose=verbose_parameter,
                                               delete_files=True, samples=x_mcs.samples)
    assert np.allclose(np.array(model_third_party_default_names.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1),
                       atol=1e-4)


def test_third_party_var_names():
    names = ['var1', 'var11', 'var111', 'var1111']
    with pytest.raises(ValueError):
        RunModel(ntasks=1, model_script='python_model_sum_scalar.py',
                 input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
                 output_script='process_third_party_output.py', output_object_name='read_output',
                 resume=False, fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True, samples=x_mcs.samples)


def test_python_serial_workflow_function_object_name_error():
    with pytest.raises(ValueError):
        model = RunModel(ntasks=1, model_script='python_model.py', vec=False, verbose=verbose_parameter)
        model.run(x_mcs.samples)


def test_python_serial_workflow_function_wrong_object_name():
    with pytest.raises(ValueError):
        model = RunModel(ntasks=1, model_script='python_model.py', vec=False, verbose=verbose_parameter,
                         model_object_name="random_model_name")
        model.run(x_mcs.samples)


def test_python_serial_workflow_function_no_objects():
    with pytest.raises(ValueError):
        model = RunModel(ntasks=1, model_script='python_model_blank.py', vec=False, verbose=verbose_parameter)
        model.run(x_mcs.samples)
