from UQpy.DimensionReduction import DirectPOD, SnapshotPOD, HOSVD
import numpy as np

# Define a dataset to test DirectPOD, SnapshotPOD and HOSVD methods
Data = np.zeros((2, 2, 3))

Data[:, :, 0] = [
[0.9073,  1.7842],
[2.1488,  4.2495]]

Data[:, :, 1] = [
[6.7121, 0.5334],
[0.3054,  0.3207]]

Data[:, :, 2] = [
[-0.3698,  0.0151],
[2.3753,  4.7146]]


def test_DirectPOD():
    pod_dir = DirectPOD(input_sol=Data, modes=1, verbose=False)
    assert round(pod_dir.run()[0][0][1][1], 6) == 0.761704


def test_SnapshotPOD():
    pod_snap = SnapshotPOD(input_sol=Data, modes=1, verbose=False)
    assert round(pod_snap.run()[0][0][1][1], 6) == -0.181528


def test_HOSVD():
    hosvd = HOSVD(input_sol=Data, reconstr_perc=90, verbose=False)
    assert round(hosvd.run(get_error=True)[0][0][1][1], 6) == 0.714928