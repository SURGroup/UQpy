from UQpy.distributions.collection import *
from UQpy.utilities.strata.Rectangular import *
from UQpy.utilities.strata.Voronoi import *
from UQpy.sampling.StratifiedSampling import *
from UQpy.utilities.strata.Delaunay import *


def test_rectangular_sts():
    marginals = [Uniform(loc=0., scale=1.), Uniform(loc=0., scale=1.)]
    strata = Rectangular(strata_number=[4, 4])
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=1, random_state=1)
    assert x.samples[6, 0] == 0.5511130624328794
    assert x.samples[12, 1] == 0.9736516658759619
    assert x.samples[2, 0] == 0.5366889727042783
    assert x.samples[9, 1] == 0.5495253722712197


def test_delaunay_sts():
    marginals = [Exponential(loc=1., scale=1.), Exponential(loc=1., scale=1.)]
    seeds = np.array([[0, 0], [0.4, 0.8], [1, 0], [1, 1]])
    strata_obj = Delaunay(seeds=seeds)
    sts_obj = StratifiedSampling(distributions=marginals, strata_object=strata_obj,
                                 samples_per_stratum_number=1, random_state=1)
    assert sts_obj.samples[2, 0] == 1.902581742436106


def test_voronoi_sts():
    marginals = [Exponential(loc=1., scale=1.), Exponential(loc=1., scale=1.)]
    strata = Voronoi(seeds_number=8, dimension=2)
    x = StratifiedSampling(distributions=marginals, strata_object=strata,
                           samples_per_stratum_number=3, random_state=3)
    assert x.samples[7, 0] == 3.6928440862661223
    assert x.samples[20, 1] == 1.1555963246730931
    assert x.samples[1, 0] == 1.8393015015282757
    assert x.samples[15, 1] == 2.117727620746169
