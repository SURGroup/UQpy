import numpy as np

# Define a dataset to test DirectPOD, SnapshotPOD and HOSVD methods
import pytest
from beartype.roar import BeartypeCallHintPepParamException

from UQpy.dimension_reduction.pod.DirectPOD import DirectPOD
from UQpy.dimension_reduction.pod.SnapshotPOD import SnapshotPOD
from UQpy.dimension_reduction.hosvd.HigherOrderSVD import HigherOrderSVD

Data = np.zeros((2, 2, 3))

Data[:, :, 0] = [
    [0.9073, 1.7842],
    [2.1488, 4.2495]]

Data[:, :, 1] = [
    [6.7121, 0.5334],
    [0.3054, 0.3207]]

Data[:, :, 2] = [
    [-0.3698, 0.0151],
    [2.3753, 4.7146]]


def test_DirectPOD_listData():
    list_data = list(Data)
    pod_dir = DirectPOD(solution_snapshots=list_data)
    reconstructed_solutions, reduced_solutions = pod_dir.run()
    actual_result = reconstructed_solutions[0][1][1]
    expected_result = 0.3054
    assert expected_result == round(actual_result, 6)


def test_DirectPOD():
    pod_dir = DirectPOD(solution_snapshots=Data, modes=1)
    assert round(pod_dir.run()[0][0][1][1], 6) == 0.761704


def test_SnapshotPOD():
    pod_snap = SnapshotPOD(solution_snapshots=Data, modes=1)
    assert round(pod_snap.run()[0][0][1][1], 6) == -0.181528


def test_SnapshotPOD_listData():
    list_data = list(Data)
    pod_dir = SnapshotPOD(solution_snapshots=list_data)
    pod_output = pod_dir.run()
    actual_result = pod_output[0][0][1][1]
    expected_result = 0.3054
    assert expected_result == round(actual_result, 6)


def test_POD_unfold():
    list_data = list(Data)
    pod_output = HigherOrderSVD.unfold3d(list_data)
    actual_result = pod_output[0][0][1]
    expected_result = 6.7121
    assert expected_result == round(actual_result, 6)


def test_HOSVD():
    hosvd = HigherOrderSVD(solution_snapshots=Data, reconstruction_percentage=90)
    reconstructed_solutions = HigherOrderSVD.reconstruct(hosvd.u1, hosvd.u2, hosvd.u3hat, hosvd.s3hat)
    assert round(reconstructed_solutions[0][1][1], 6) == 0.714928


def test_DirectPOD_modes_less_than_zero():
    with pytest.raises(BeartypeCallHintPepParamException):
        pod_dir = DirectPOD(solution_snapshots=Data, modes=-1)


def test_DirectPOD_mode_non_integer():
    with pytest.raises(BeartypeCallHintPepParamException):
        pod_dir = DirectPOD(solution_snapshots=Data, modes=1.5)


def test_DirectPOD_reconstr_perc_less_than_zero():
    with pytest.raises(BeartypeCallHintPepParamException):
        pod_dir = DirectPOD(solution_snapshots=Data, reconstruction_percentage=-1)


def test_DirectPOD_both_modes_and_reconstr_error():
    with pytest.raises(ValueError):
        pod_dir = DirectPOD(solution_snapshots=Data, modes=1, reconstruction_percentage=50)


def test_HOSVD_modes_less_than_zero():
    with pytest.raises(BeartypeCallHintPepParamException):
        hosvd = HigherOrderSVD(solution_snapshots=Data, modes=-1)


def test_HOSVD_mode_non_integer():
    with pytest.raises(BeartypeCallHintPepParamException):
        hosvd = HigherOrderSVD(solution_snapshots=Data, modes=1.5)


def test_HOSVD_reconstr_perc_less_than_zero():
    with pytest.raises(BeartypeCallHintPepParamException):
        hosvd = HigherOrderSVD(solution_snapshots=Data, reconstruction_percentage=-1)


def test_HOSVD_both_modes_and_reconstr_error():
    with pytest.raises(ValueError):
        hosvd = HigherOrderSVD(solution_snapshots=Data, modes=1, reconstruction_percentage=50)


def test_SnapshotPOD_modes_less_than_zero():
    with pytest.raises(BeartypeCallHintPepParamException):
        snap = SnapshotPOD(solution_snapshots=Data, modes=-1)


def test_SnapshotPOD_mode_non_integer():
    with pytest.raises(BeartypeCallHintPepParamException):
        snap = SnapshotPOD(solution_snapshots=Data, modes=1.5)


def test_SnapshotPOD_reconstr_perc_less_than_zero():
    with pytest.raises(BeartypeCallHintPepParamException):
        snap = SnapshotPOD(solution_snapshots=Data, reconstruction_percentage=-1)


def test_SnapshotPOD_both_modes_and_reconstr_error():
    with pytest.raises(ValueError):
        snap = SnapshotPOD(solution_snapshots=Data, modes=1, reconstruction_percentage=50)
