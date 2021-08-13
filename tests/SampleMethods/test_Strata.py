from UQpy.SampleMethods import RectangularStrata, VoronoiStrata, DelaunayStrata
import pytest
import numpy as np
import os

strata = RectangularStrata(nstrata=[3, 3], verbose=True)
dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, 'strata.txt')
strata1 = RectangularStrata(input_file=filepath)
fig = strata1.plot_2d()

strata_vor = VoronoiStrata(nseeds=8, dimension=2, random_state=3, verbose=True)
strata_vor1 = VoronoiStrata(nseeds=8, dimension=2, niters=0, random_state=3, verbose=True)
seeds_ = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
strata_vor2 = VoronoiStrata(seeds=seeds_, dimension=3, niters=0, random_state=3, verbose=True)

seeds = np.array([[0, 0], [0.4, 0.8], [1, 0], [1, 1]])
strata_del = DelaunayStrata(seeds=seeds, dimension=2, verbose=True)
strata_del1 = DelaunayStrata(nseeds=4, dimension=2, random_state=10)


# Unit tests
def test_1():
    """
    Test output attributes of strata object.
    """
    tmp1 = (np.round(strata.seeds, 3) == np.array([[0., 0.], [0.333, 0.], [0.667, 0.], [0., 0.333], [0.333, 0.333],
                                                   [0.667, 0.333], [0., 0.667], [0.333, 0.667], [0.667, 0.667]])).all()
    tmp2 = (np.round(strata.widths, 3) == np.array([[0.333, 0.333], [0.333, 0.333], [0.333, 0.333], [0.333, 0.333],
                                                    [0.333, 0.333], [0.333, 0.333], [0.333, 0.333], [0.333, 0.333],
                                                    [0.333, 0.333]])).all()
    tmp3 = (np.round(strata.volume, 3) == np.array([0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111,
                                                    0.111])).all()
    assert tmp1 and tmp2 and tmp3


def test_2():
    """
    Test output attributes of strata1 object.
    """
    tmp1 = (np.round(strata1.seeds, 3) == np.array([[0., 0.], [0., 0.333], [0., 0.667], [0.5, 0.], [0.5, 0.5],
                                                    [0.75, 0.5]])).all()
    tmp2 = (np.round(strata1.widths, 3) == np.array([[0.5, 0.333], [0.5, 0.333], [0.5, 0.333], [0.5, 0.5], [0.25, 0.5],
                                                     [0.25, 0.5]])).all()
    tmp3 = (np.round(strata1.volume, 3) == np.array([0.167, 0.167, 0.167, 0.25, 0.125, 0.125])).all()
    assert tmp1 and tmp2 and tmp3


def test_3():
    """
        Test error check.
    """
    with pytest.raises(TypeError):
        RectangularStrata(random_state='ab')


def test_rectangular_no_attributes():
    """
        No attribute is assigned to define the strata.
    """
    with pytest.raises(RuntimeError):
        RectangularStrata()


def test_not_space_filling():
    """
        No attribute is assigned to define the strata.
    """
    with pytest.raises(RuntimeError):
        RectangularStrata(input_file='strata1.txt')


def test_over_filling():
    """
        No attribute is assigned to define the strata.
    """
    with pytest.raises(RuntimeError):
        RectangularStrata(input_file='strata2.txt')


def test_5():
    """
    Test output attributes of strata_vor object.
    """
    tmp1 = (np.round(strata_vor.seeds, 3) == np.array([[0.464, 0.82], [0.271, 0.599], [0.855, 0.85], [0.197, 0.159],
                                                       [0.1, 0.416], [0.05, 0.701], [0.704, 0.211],
                                                       [0.765, 0.557]])).all()
    tmp2 = (np.round(strata_vor.volume, 3) == np.array([0.161, 0.158, 0.097, 0.128, 0.032, 0.054, 0.249, 0.122])).all()
    assert tmp1 and tmp2


def test_6():
    """
    Test output attributes of strata_vor2 object.
    """
    assert (np.round(strata_vor2.volume, 2) == np.array([1., 1., 1., 1., 1., 1., 1., 1.])).all()


def test_voronoi_volume3():
    """
    Test output attributes of strata_vor1 object.
    """
    tmp1 = (np.round(strata_vor1.seeds, 3) == np.array([[0.551, 0.708], [0.291, 0.511], [0.893, 0.896], [0.126, 0.207],
                                                        [0.051, 0.441], [0.03, 0.457], [0.649, 0.278],
                                                        [0.676, 0.591]])).all()
    tmp2 = (np.round(strata_vor1.volume, 3) == np.array([0.161, 0.158, 0.097, 0.128, 0.032, 0.054, 0.249, 0.122])).all()
    assert tmp1 and tmp2


def test_7():
    """
    Test output attributes of strata_vor1 object.
    """
    tmp1 = (np.round(strata_del.centroids, 3) == np.array([[0.133, 0.6], [0.467, 0.267], [0.467, 0.933],
                                                           [0.8, 0.6]])).all()
    tmp2 = (np.round(strata_del.volume, 3) == np.array([0.2, 0.4, 0.1, 0.3])).all()
    assert tmp1 and tmp2

