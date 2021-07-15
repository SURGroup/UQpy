import numpy as np
import pytest
from UQpy.Distributions import Normal, MVNormal
from UQpy.SampleMethods import MCS

dist1 = Normal(loc=0., scale=1.)
dist2 = Normal(loc=0., scale=1.)

x = MCS(dist_object=dist1, nsamples=5, random_state=np.random.RandomState(123),  verbose=True)
x.transform_u01()
y = MCS(dist_object=[dist1, dist2], verbose=True)
y.run(nsamples=5, random_state=123)
y.transform_u01()

# Call run method multiple time, to cover lines where samples are append to existing ones
z1 = MCS(dist_object=dist1, nsamples=2, random_state=123,  verbose=True)
z1.run(nsamples=2)
z2 = MCS(dist_object=[dist1, dist2], nsamples=2, random_state=np.random.RandomState(123),  verbose=True)
z2.run(nsamples=2)
# Same object as z2, just to cover lines where, random_state is an integer
z3 = MCS(dist_object=[dist1, dist2], nsamples=2, random_state=123,  verbose=True)

z4 = MCS(dist_object=[MVNormal([0, 0])], nsamples=2, random_state=np.random.RandomState(123),  verbose=True)
z4.run(nsamples=2)
z4.transform_u01()
dist3 = Normal(loc=0., scale=1.)
del dist3.rvs
z5 = MCS(dist_object=[dist3], random_state=np.random.RandomState(123),  verbose=True)


def test_dist_object():
    """Validate dist_object, when dist_object is a distribution object."""
    with pytest.raises(TypeError):
        MCS(dist_object='abc', random_state=np.random.RandomState(123),  verbose=True)


def test_dist_object2():
    """Validate dist_object, when dist_object is not a list of distribution object."""
    with pytest.raises(TypeError):
        MCS(dist_object=['abc'], random_state=np.random.RandomState(123),  verbose=True)


def test_dist_object3():
    """Create a MCS object using DistributionND class object. Validate dist_object, when dist_object is not a
    list of distribution object."""
    with pytest.raises(TypeError):
        MCS(dist_object=['abc'], random_state=np.random.RandomState(123),  verbose=True)


def test_distribution_nd():
    """Validate error check, when distribution object doesn't have 'rvs' method."""
    with pytest.raises(ValueError):
        z5.run(nsamples=2)


def test_samples1():
    """ Check the samples attribute, when dist_object is a distribution class object."""
    expected_samples = np.array([[-1.09], [1.], [0.28], [-1.51], [-0.58]])
    assert (x.samples.round(2) == expected_samples).all()


def test_samples2():
    """ Check the samples attribute, when dist_object is list of multiple distribution class object."""
    expected_samples = np.array([[-1.09,  1.65], [1., -2.43], [0.28, -0.43], [-1.51,  1.27], [-0.58, -0.87]])
    assert (y.samples.round(2) == expected_samples).all()


def test_samples3():
    """ Check the samples attribute, when 'run' method is called twice and samples are appended, and dist_object is a
    distribution class object.."""
    expected_samples = np.array([[-1.09], [1.], [0.28], [-1.51]])
    assert (z1.samples.round(2) == expected_samples).all()


def test_samples4():
    """ Check the samples attribute, when 'run' method is called twice and samples are appended, and dist_object is a
    list of multiple distribution class object."""
    expected_samples = np.array([[-1.09,  0.28], [1., -1.51], [-0.58, -2.43], [1.65, -0.43]])
    assert (z2.samples.round(2) == expected_samples).all()


def test_samples5():
    """ Check the samples attribute, when dist_object is a list of distributionND class object."""
    expected_samples = np.array([[[-1.09,  1.]], [[0.28, -1.51]], [[-0.58,  1.65]], [[-2.43, -0.43]]])
    assert (z4.samples.round(2) == expected_samples).all()


def test_samples_u01():
    """ Check the samplesU01 attribute, when dist_object is a distribution class object."""
    expected_samples_u01 = np.array([[0.14], [0.84], [0.61], [0.07], [0.28]])
    assert (x.samplesU01.round(2) == expected_samples_u01).all()


def test_samples2_u01():
    """ Check the samplesU01 attribute, when dist_object is list of multiple distribution class object."""
    expected_samples_u01 = np.array([[0.14, 0.95], [0.84, 0.01], [0.61, 0.33], [0.07, 0.9], [0.28, 0.19]])
    assert (y.samplesU01.round(2) == expected_samples_u01).all()


def test_samples2_u03():
    """ Check the samplesU01 attribute, when dist_object is list of distributionND class objects."""
    assert [x_.round(2)[0, 0] for x_ in z4.samplesU01] == [0.12, 0.04, 0.27, 0.0]


def test_nsamples_none():
    """Validate error check, when nsamples is None, while calling 'run' method."""
    with pytest.raises(ValueError):
        MCS(dist_object=dist1, random_state=np.random.RandomState(123),  verbose=True).run(nsamples=None)


def test_nsamples_not_integer():
    """Validate error check, when nsamples is not an integer, while calling 'run' method."""
    with pytest.raises(ValueError):
        MCS(dist_object=dist1, random_state=np.random.RandomState(123),  verbose=True).run(nsamples='abc')


def test_random_state1():
    """Check if random_state attribute is an integer, np.random.RandomState object or None, when dist_object is a
    distribution class object."""
    with pytest.raises(TypeError):
        MCS(dist_object=dist1, random_state='abc')


def test_random_state2():
    """Check if random_state attribute is not an integer, np.random.RandomState object or None, when dist_object is a
    list of multiple distribution class object."""
    with pytest.raises(TypeError):
        MCS(dist_object=[dist1, dist2], random_state='abc')


def test_run_random_state():
    """Check if random_state attribute is not an integer, np.random.RandomState object or None, when when 'run' method
    is called."""
    with pytest.raises(TypeError):
        MCS(dist_object=dist1).run(nsamples=5, random_state='abc')


def test_cdf_method1():
    """Check, if dist_object attribute has 'cdf' method, when dist_object is a distribution class object."""
    del x.dist_object.cdf
    with pytest.raises(ValueError):
        x.transform_u01()


def test_cdf_method2():
    """Check, if dist_object attribute has 'cdf' method,  when dist_object is list of multiple distribution class
    object."""
    y.dist_object, y.array = [1, 1], True
    with pytest.raises(ValueError):
        y.transform_u01()


def test_cdf_method3():
    """Check, if dist_object attribute has 'cdf' method,  when dist_object is a distributionND class object."""

    z4.dist_object, z4.list = [1, 2], True
    with pytest.raises(ValueError):
        z4.transform_u01()

