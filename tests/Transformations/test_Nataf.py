# Test the Nataf transformation

from UQpy.Distributions import *
from UQpy.Transformations import Nataf
import numpy as np
import pytest


def test_type_of_dist1():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    assert ntf_obj.dimension == 2


def test_type_of_dist2():
    dist1 = Normal(loc=0.0, scale=1.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(Exception):
        assert Nataf(dist_object=[dist1, "Beta"], corr_x=rx)


def test_type_of_dist3():
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(Exception):
        assert Nataf(dist_object="Normal", corr_x=rx)


def test_corr1():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    ntf_obj = Nataf(dist_object=[dist1, dist2])
    assert np.all(np.equal(ntf_obj.corr_x, np.eye(2)))


def test_corr2():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    ntf_obj = Nataf(dist_object=[dist1, dist2])
    assert np.all(np.equal(ntf_obj.corr_z, np.eye(2)))


def test_corr_x1():
    dist1 = Uniform(loc=0.0, scale=1.0)
    dist2 = Uniform(loc=0.0, scale=1.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    assert np.all(np.equal(ntf_obj.corr_z, rx))


def test_corr_x2():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    assert np.all(np.equal(ntf_obj.corr_z, rx))


def test_corr_x3():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    rx = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    assert np.all(np.equal(ntf_obj.corr_z, rx))


def test_corr_x4():
    dist1 = Uniform(loc=0.0, scale=1.0)
    dist2 = Uniform(loc=0.0, scale=1.0)
    rx = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_x=rx)
    np.testing.assert_allclose(
        ntf_obj.corr_z,
        [[1.0, 0.8134732861515996], [0.8134732861515996, 1.0]],
        rtol=1e-09,
    )


def test_corr_z1():
    dist1 = Uniform(loc=0.0, scale=1.0)
    dist2 = Uniform(loc=0.0, scale=1.0)
    rz = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_z=rz)
    assert np.all(np.equal(ntf_obj.corr_x, rz))


def test_corr_z2():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    rz = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_z=rz)
    assert np.all(np.equal(ntf_obj.corr_x, rz))


def test_corr_z3():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_z=rz)
    assert np.all(np.equal(ntf_obj.corr_x, rz))


def test_corr_z4():
    dist1 = Uniform(loc=0.0, scale=1.0)
    dist2 = Uniform(loc=0.0, scale=1.0)
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_z=rz)
    assert (
        ntf_obj.corr_x == [[1.0, 0.7859392826067285], [0.7859392826067285, 1.0]]
    ).all()


def test_h():
    dist1 = Normal(loc=0.0, scale=1.0)
    dist2 = Normal(loc=0.0, scale=1.0)
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1, dist2], corr_z=rz)
    np.testing.assert_allclose(ntf_obj.H, [[1.0, 0.0], [0.8, 0.6]], rtol=1e-09)


def test_samples_x():
    dist1 = Normal(loc=0.0, scale=1.0)
    ntf_obj = Nataf(dist_object=[dist1])
    assert ntf_obj.samples_x is None


def test_samples_z():
    dist1 = Normal(loc=0.0, scale=1.0)
    ntf_obj = Nataf(dist_object=[dist1])
    assert ntf_obj.samples_z is None


def test_samples_x_shape():
    dist1 = Normal(loc=0.0, scale=5.0)
    samples_x = np.array([0.3, 1.2, 3.5])
    ntf_obj = Nataf(dist_object=[dist1])
    ntf_obj.run(samples_x)
    assert ntf_obj.samples_x.shape == (3, 1)


def test_samples_z_shape():
    dist1 = Normal(loc=0.0, scale=5.0)
    ntf_obj = Nataf(dist_object=[dist1])
    samples_z = np.array([0.3, 1.2, 3.5])
    ntf_obj.run(samples_z)
    assert ntf_obj.samples_z.shape == (3, 1)


def test_samples_x_jxz1():
    dist1 = Normal(loc=0.0, scale=5.0)
    ntf_obj = Nataf(dist_object=[dist1])
    samples_x = np.array([0.3, 1.2, 3.5])
    ntf_obj.run(samples_x=samples_x, jacobian=False)
    assert ntf_obj.jxz is None


def test_samples_x_jxz2():
    dist1 = Uniform(loc=0.0, scale=5.0)
    dist2 = Uniform(loc=0.0, scale=3.0)
    ntf_obj = Nataf(dist_object=[dist1, dist2])
    samples_x = np.array([[0.3, 1.2, 3.5], [0.2, 2.4, 0.9]]).T
    ntf_obj.run(samples_x=samples_x, jacobian=True)
    g = []
    for i in range(3):
        if i == 0:
            g.append(
                (
                    ntf_obj.jxz[i]
                    == np.array([[1.6789373877365803, 0.0], [0.0, 2.577850090371836]])
                ).all()
            )
        elif i == 1:
            g.append(
                (
                    ntf_obj.jxz[i]
                    == np.array([[0.6433491348614259, 0.0], [0.0, 1.1906381155257868]])
                ).all()
            )
        else:
            g.append(
                (
                    ntf_obj.jxz[i]
                    == np.array([[0.5752207318528584, 0.0], [0.0, 0.958701219754764]])
                ).all()
            )
    assert np.all(g)


def test_samples_x1():
    dist1 = Uniform(loc=0.0, scale=5.0)
    dist2 = Uniform(loc=0.0, scale=3.0)
    ntf_obj = Nataf(dist_object=[dist1, dist2])
    samples_x = np.array([[0.3, 1.2, 3.5], [0.2, 2.4, 0.9]]).T
    ntf_obj.run(samples_x=samples_x, jacobian=True)
    g = []
    for i in range(3):
        if i == 0:
            g.append(
                (
                    ntf_obj.samples_z[i]
                    == np.array([-1.5547735945968535, -1.501085946044025])
                ).all()
            )
        elif i == 1:
            g.append(
                (
                    ntf_obj.samples_z[i]
                    == np.array([-0.7063025628400874, 0.841621233572914])
                ).all()
            )
        else:
            g.append(
                (
                    ntf_obj.samples_z[i]
                    == np.array([0.5244005127080407, -0.5244005127080409])
                ).all()
            )
    assert np.all(g)


def test_samples_z_jzx1():
    dist1 = Normal(loc=0.0, scale=5.0)
    ntf_obj = Nataf(dist_object=[dist1])
    samples_z = np.array([0.3, 1.2, 3.5])
    ntf_obj.run(samples_z=samples_z, jacobian=False)
    assert ntf_obj.jzx is None


def test_samples_z_jzx2():
    dist1 = Uniform(loc=0.0, scale=5.0)
    dist2 = Uniform(loc=0.0, scale=3.0)
    ntf_obj = Nataf(dist_object=[dist1, dist2])
    samples_z = np.array([[0.3, 1.2], [0.2, 2.4]]).T
    ntf_obj.run(samples_z=samples_z, jacobian=True)
    g = []
    for i in range(2):
        if i == 0:
            g.append(
                (
                    ntf_obj.jzx[i]
                    == np.array([[0.524400601939789, 0.0], [0.0, 0.8524218415758338]])
                ).all()
            )
        else:
            g.append(
                (
                    ntf_obj.jzx[i]
                    == np.array([[1.0299400748281828, 0.0], [0.0, 14.884586948005541]])
                ).all()
            )
    assert np.all(g)


def test_samples_z2():
    dist1 = Uniform(loc=0.0, scale=5.0)
    dist2 = Uniform(loc=0.0, scale=3.0)
    ntf_obj = Nataf(dist_object=[dist1, dist2])
    samples_z = np.array([[0.3, 1.2], [0.2, 2.4]]).T
    ntf_obj.run(samples_z=samples_z, jacobian=True)
    g = []
    for i in range(2):
        if i == 0:
            g.append(
                (
                    ntf_obj.samples_x[i]
                    == np.array([3.089557110944763, 1.737779128317309])
                ).all()
            )
        elif i == 1:
            g.append(
                (
                    ntf_obj.samples_x[i]
                    == np.array([4.424651648891459, 2.9754073922262116])
                ).all()
            )
    assert np.all(g)


def test_itam_1():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx)
    assert ntf_obj.itam_beta == 1.0


def test_itam_2():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx)
    assert ntf_obj.itam_max_iter == 100


def test_itam_3():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx)
    assert ntf_obj.itam_threshold1 == 0.001


def test_itam_4():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx)
    assert ntf_obj.itam_threshold2 == 0.1


def test_itam_1a():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx, itam_beta=2.0)
    assert ntf_obj.itam_beta == 2.0


def test_itam_2a():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx, itam_max_iter=200)
    assert ntf_obj.itam_max_iter == 200


def test_itam_2m():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx, itam_max_iter=10.5)
    assert ntf_obj.itam_max_iter == 10


def test_itam_3a():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx, itam_threshold1=0.002)
    assert ntf_obj.itam_threshold1 == 0.002


def test_itam_4a():
    dist1 = Uniform(loc=0.0, scale=5.0)
    rx = np.array([[1.0, 0.0], [0.0, 1.0]])
    ntf_obj = Nataf(dist_object=[dist1], corr_x=rx, itam_threshold2=0.3)
    assert ntf_obj.itam_threshold2 == 0.3


def test_distortion_z2x_finite_moments():
    dist1 = Lognormal(s=0.0, loc=0.0, scale=1.0)
    dist2 = Uniform(loc=0.0, scale=1.0)
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    with pytest.raises(Exception):
        assert Nataf(dist_object=[dist1, dist2], corr_z=rz)


def distortion_z2x_dist_object():
    dist1 = Lognormal(s=0.0, loc=0.0, scale=1.0)
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    with pytest.raises(Exception):
        assert Nataf.distortion_z2x(dist_object=[dist1, "Beta"], corr_z=rz)
