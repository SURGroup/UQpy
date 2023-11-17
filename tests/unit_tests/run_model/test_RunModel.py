import shutil

from beartype.roar import BeartypeCallHintPepParamException

from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.run_model import ThirdPartyModel
from UQpy.sampling import MonteCarloSampling
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions import Normal
import pytest
import numpy as np

d = Normal(loc=0, scale=1)
x_mcs = MonteCarloSampling(distributions=[d, d, d], nsamples=5, random_state=1234)
x_mcs_new = MonteCarloSampling(distributions=[d, d, d], nsamples=5, random_state=2345)
verbose_parameter = True


# def test_div_zero():
#     with pytest.raises(TypeError):
#         model = PythonModel(model_script='python_model.py', model_object_name='SumRVs', fmt=20,
#                             delete_files=True)
#         runmodel_object = RunModel_New(model=model)
#
#
# def test_fmt_1():
#     with pytest.raises(TypeError):
#         model = PythonModel(model_script='python_model.py', model_object_name='SumRVs', fmt=20,
#                             delete_files=True)
#
#
# def test_fmt_2():
#     with pytest.raises(ValueError):
#         model = PythonModel(model_script='python_model.py', model_object_name='SumRVs', fmt="random_string",
#                             delete_files=True)
#         runmodel_object = RunModel_New(model=model)


def test_var_names():
    with pytest.raises(BeartypeCallHintPepParamException):
        model = PythonModel(model_script='python_model.py', model_object_name='SumRVs', var_names=[20],
                            delete_files=True)
        runmodel_object = RunModel(model=model)



def test_model_script():
    with pytest.raises(ValueError):
        model = PythonModel(model_script='random_file_name', model_object_name='SumRVs', delete_files=True)
        runmodel_object = RunModel(model=model)


def test_samples():
    with pytest.raises(BeartypeCallHintPepParamException):
        model = PythonModel(model_script='python_model.py', model_object_name='SumRVs',
                         delete_files=True)
        runmodel_object = RunModel(model=model, samples="samples_string")



def test_python_serial_workflow_class_vectorized():
    model = PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    model_python_serial_class = RunModel(model=model)
    model_python_serial_class.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))



def test_python_serial_workflow_class():
    model = PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    model_python_serial_class = RunModel(model=model)
    model_python_serial_class.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_direct_samples():
    model = PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    model_python_serial_class = RunModel(model=model, samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_append_samples_true():
    model = PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    model_python_serial_class = RunModel(model=model, samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))
    model_python_serial_class.run(x_mcs_new.samples, append_samples=True)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(),
                       np.sum(np.vstack((x_mcs.samples, x_mcs_new.samples)), axis=1))


def test_append_samples_false():
    model = PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    model_python_serial_class = RunModel(model=model, samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))
    model_python_serial_class.run(x_mcs_new.samples, append_samples=False)
    assert np.allclose(np.array(model_python_serial_class.qoi_list).flatten(), np.sum(x_mcs_new.samples, axis=1))


def test_python_serial_workflow_function_vectorized():
    model = PythonModel(model_script='python_model.py', model_object_name='sum_rvs')
    model_python_serial_function = RunModel(model=model)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


def test_python_serial_workflow_function():
    model = PythonModel(model_script='python_model.py', model_object_name='sum_rvs')
    model_python_serial_function = RunModel(model=model)
    model_python_serial_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))


# def test_python_serial_workflow_function_no_object_name():
#     model_python_serial_function = RunModel(ntasks=1, model_script='python_model_function.py', vec=False)
#     model_python_serial_function.run(samples=x_mcs.samples)
#     assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))
#     shutil.rmtree(model_python_serial_function.model_dir)


# def test_python_serial_workflow_class_no_object_name():
#     model_python_serial_function = RunModel(ntasks=1, model_script='python_model_class.py', vec=False)
#     model_python_serial_function.run(samples=x_mcs.samples)
#     assert np.allclose(np.array(model_python_serial_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))

@pytest.mark.skip()
def test_python_parallel_workflow_class():
    model = PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    model_python_parallel_class = RunModel(model=model, samples=x_mcs.samples, ntasks=3)
    model_python_parallel_class.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_parallel_class.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))
    shutil.rmtree(model_python_parallel_class.model_dir)

@pytest.mark.skip()
def test_python_parallel_workflow_function():
    model = PythonModel(model_script='python_model.py', model_object_name='sum_rvs')
    model_python_parallel_function = RunModel(model=model, ntasks=3)
    model_python_parallel_function.run(samples=x_mcs.samples)
    assert np.allclose(np.array(model_python_parallel_function.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1))
    shutil.rmtree(model_python_parallel_function.model_dir)


# def test_third_party_serial():
#     names = ['var1', 'var11', 'var111']
#     model = ThirdPartyModel(model_script='python_model_sum_scalar.py',
#                             input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
#                             output_script='process_third_party_output.py', output_object_name='read_output',
#                             fmt="{:>10.4f}", delete_files=True)
#     m = RunModel_New(model=model)
#     m.run(x_mcs.samples)
#     assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)
#     shutil.rmtree(m.model.model_dir)


# def test_third_party_serial_output_class():
#     names = ['var1', 'var11', 'var111']
#     m = RunModel(ntasks=1, model_script='python_model_sum_scalar.py',
#                  input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
#                  output_script='process_third_party_output_class.py', output_object_name='ReadOutput',
#                  resume=False, fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True)
#     m.run(x_mcs.samples)
#     assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)
#     shutil.rmtree(m.model_dir)


# def test_third_party_serial_no_output_class():
#     names = ['var1', 'var11', 'var111']
#     m = RunModel(ntasks=1, model_script='python_model_sum_scalar.py', input_template='sum_scalar.py', var_names=names,
#                  model_object_name="matlab", output_script='process_third_party_output_class.py', resume=False,
#                  fmt="{:>10.4f}", verbose=verbose_parameter, delete_files=True)
#     m.run(x_mcs.samples)
#     assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)
#     shutil.rmtree(m.model_dir)


# def test_third_party_serial_no_output_function():
#     names = ['var1', 'var11', 'var111']
#     model = ThirdPartyModel(model_script='python_model_sum_scalar.py',
#                             input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
#                             output_script='process_third_party_output.py', output_object_name='read_output',
#                             fmt="{:>10.4f}", delete_files=True)
#     m = RunModel_New(model=model)
#     m.run(x_mcs.samples)
#     assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)
#     shutil.rmtree(m.model_dir)


@pytest.mark.skip()
def test_third_party_parallel():
    names = ['var1', 'var11', 'var111']
    model = ThirdPartyModel(model_script='python_model_sum_scalar.py', fmt="{:>10.4f}", delete_files=True,
                 input_template='sum_scalar.py', var_names=names, model_object_name="matlab",
                 output_script='process_third_party_output.py', output_object_name='read_output')
    m = RunModel(model=model, ntasks=3)
    m.run(x_mcs.samples)
    assert np.allclose(np.array(m.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1), atol=1e-4)
    shutil.rmtree(m.model.model_dir)


@pytest.mark.skip()
def test_third_party_default_var_names():
    model = ThirdPartyModel(model_script='python_model_sum_scalar.py', fmt="{:>10.4f}", delete_files=True,
                            input_template='sum_scalar.py', model_object_name="matlab",
                            output_script='process_third_party_output.py', output_object_name='read_output')
    model_third_party_default_names = RunModel(model=model, ntasks=3, samples=x_mcs.samples)
    assert np.allclose(np.array(model_third_party_default_names.qoi_list).flatten(), np.sum(x_mcs.samples, axis=1),
                       atol=1e-4)
    shutil.rmtree(model_third_party_default_names.model.model_dir)


def test_third_party_var_names():
    names = ['var1', 'var11', 'var111', 'var1111']
    with pytest.raises(TypeError):
        model = ThirdPartyModel(model_script='python_model_sum_scalar.py', fmt="{:>10.4f}", delete_files=True,
                                input_template='sum_scalar.py', model_object_name="matlab",
                                output_script='process_third_party_output.py', output_object_name='read_output')
        model_third_party_default_names = RunModel(model=model, ntasks=3, samples=x_mcs.samples)


def test_python_serial_workflow_function_object_name_error():
    with pytest.raises(TypeError):
        model = PythonModel(model_script='python_model.py')
        model = RunModel(model=model)
        model.run(x_mcs.samples)


def test_python_serial_workflow_function_wrong_object_name():
    with pytest.raises(AttributeError):
        model = PythonModel(model_script='python_model.py', model_object_name="random_model_name")
        model = RunModel(model=model)
        model.run(x_mcs.samples)


def test_python_serial_workflow_function_no_objects():
    with pytest.raises(TypeError):
        model = PythonModel(model_script='python_model_blank.py')
        model = RunModel(model=model)
        model.run(x_mcs.samples)
