"""
Tests for diffusion module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import warnings
import numpy as np
from tqdm import tqdm
from numpy.testing import assert_almost_equal, assert_equal
from kinisi.diffusion import Bootstrap, MSDBootstrap, MSTDBootstrap, MSCDBootstrap, _bootstrap
from uravu.distribution import Distribution

RNG = np.random.RandomState(43)


class TestBootstrap(unittest.TestCase):
    """
    Tests for the diffusion.Bootstrap class.
    """

    def test_dictionary_roundtrip(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        n_o = np.ones(len(disp_3d)) * 100
        dt = np.linspace(100, 1000, 10)
        a = Bootstrap(dt, disp_3d, n_o)
        b = Bootstrap.from_dict(a.to_dict())
        for i, d in enumerate(disp_3d):
            assert_almost_equal(a._displacements[i], b._displacements[i])
        assert_almost_equal(a._delta_t, b._delta_t)
        assert a._max_obs == b._max_obs
        assert a._distributions == b._distributions
        assert a.dt.size == b.dt.size
        assert isinstance(b.dt, np.ndarray)
        assert b._diffusion_coefficient is None
        assert b.intercept is None

    def test_initialisation(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        n_o = np.ones(len(disp_3d)) * 100
        dt = np.linspace(100, 1000, 10)
        bs = Bootstrap(dt, disp_3d, n_o)
        for i, d in enumerate(disp_3d):
            assert_almost_equal(bs._displacements[i], d)
        assert_almost_equal(bs._delta_t, dt)
        assert bs._max_obs == 20
        assert bs._distributions == []
        assert bs.dt.size == 0
        assert isinstance(bs.dt, np.ndarray)
        assert bs._diffusion_coefficient is None
        assert bs.intercept is None

    def test_initialisation_sub_sample_dt(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        n_o = np.ones(len(disp_3d)) * 100
        dt = np.linspace(100, 1000, 10)
        bs = Bootstrap(dt, disp_3d, n_o, sub_sample_dt=2)
        for i, d in enumerate(disp_3d[::2]):
            assert_almost_equal(bs._displacements[i], d)
        assert_almost_equal(bs._delta_t, dt[::2])
        assert bs._max_obs == 20
        assert bs._distributions == []
        assert bs.dt.size == 0
        assert isinstance(bs.dt, np.ndarray)
        assert bs._diffusion_coefficient is None
        assert bs.intercept is None

    def test_iterator_true(self):
        result = Bootstrap.iterator(True, range(10))
        assert isinstance(result, tqdm)

    def test_iterator_false(self):
        result = Bootstrap.iterator(False, range(10))
        assert isinstance(result, range)

    def test_sample_until_normal(self):
        distro1 = Bootstrap.sample_until_normal(RNG.normal(0, 1, size=2000),
                                                50,
                                                1000,
                                                100000,
                                                random_state=np.random.RandomState(1))
        assert distro1.size == 1000

    def test_sample_until_normal_random(self):
        with warnings.catch_warnings(record=True) as _:
            distro1 = Bootstrap.sample_until_normal(np.arange(1, 10, 1),
                                                    5,
                                                    100,
                                                    100,
                                                    random_state=np.random.RandomState(0))
            distro2 = Bootstrap.sample_until_normal(np.arange(1, 10, 1),
                                                    5,
                                                    100,
                                                    100,
                                                    random_state=np.random.RandomState(0))
            assert_almost_equal(distro1.samples, distro2.samples)

    def test_ngp_calculation(self):
        result = Bootstrap.ngp_calculation(np.array([1, 2, 3]))
        assert_almost_equal(result, -0.3)


class TesMSTDBootstrap(unittest.TestCase):
    """
    Tests for the diffusion.MSDBootstrap class.
    """

    def test_dictionary_roundtrip(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            a = MSDBootstrap(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            b = MSDBootstrap.from_dict(a.to_dict())
            for i, d in enumerate(disp_3d):
                assert_almost_equal(a._displacements[i], b._displacements[i])
            assert_almost_equal(a._delta_t, b._delta_t)
            assert a._max_obs == b._max_obs
            assert_equal(a.n, b.n)
            assert_equal(a.s, b.s)
            assert_equal(a.v, b.v)
            assert_equal(a.ngp, b.ngp)
            for i, d in enumerate(a._distributions):
                assert_almost_equal(d.samples, b._distributions[i].samples)
            for i, d in enumerate(a.euclidian_displacements):
                assert_equal(d.samples, b.euclidian_displacements[i].samples)
            assert a.dt.size == b.dt.size
            assert isinstance(b.dt, np.ndarray)
            assert b._diffusion_coefficient is None
            assert b.intercept is None

    def test_initialisation(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            assert bs.n.shape == (10, )
            assert bs.s.shape == (10, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (10, )
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000

    def test_initialisation_n_resamples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, n_resamples=10, random_state=np.random.RandomState(0))
            assert bs.n.shape == (10, )
            assert bs.s.shape == (10, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (10, )
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 10

    def test_initialisation_max_resamples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt,
                              disp_3d,
                              n_o,
                              n_resamples=10,
                              max_resamples=100,
                              random_state=np.random.RandomState(0))
            assert bs.n.shape == (10, )
            assert bs.s.shape == (10, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (10, )
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size <= 110

    def test_initialisation_random_state(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs1 = MSDBootstrap(dt, disp_3d, n_o, bootstrap=True, random_state=np.random.RandomState(0))
            assert bs1.n.shape == (10, )
            assert bs1.s.shape == (10, )
            assert_almost_equal(bs1.v, np.square(bs1.s))
            assert bs1.ngp.shape == (10, )
            assert len(bs1.euclidian_displacements) == 10
            for i in bs1.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs1._distributions:
                assert i.samples.size >= 1000
            bs2 = MSDBootstrap(dt, disp_3d, n_o, bootstrap=True, random_state=np.random.RandomState(0))
            assert bs1._distributions[-1].size == bs2._distributions[-1].size
            assert_almost_equal(bs1._distributions[-1].samples, bs2._distributions[-1].samples)

    def test_initialisation_progress(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (10, )
            assert bs.s.shape == (10, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (10, )
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000
            assert isinstance(bs._iterator, range)

    def test_initialisation_skip_where_low_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(1, i, 3) for i in range(10, 1, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (9, )
            assert bs.s.shape == (9, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (9, )
            assert len(bs.euclidian_displacements) == 9
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000
            assert isinstance(bs._iterator, range)

    def test_bootstrap_dictionary_roundtrip(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            a = MSDBootstrap(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            a.diffusion()
            b = MSDBootstrap.from_dict(a.to_dict())
            for i, d in enumerate(disp_3d):
                assert_almost_equal(a._displacements[i], b._displacements[i])
            assert_almost_equal(a._delta_t, b._delta_t)
            assert a._max_obs == b._max_obs
            assert_equal(a.n, b.n)
            assert_equal(a.s, b.s)
            assert_equal(a.v, b.v)
            assert_equal(a.ngp, b.ngp)
            for i, d in enumerate(a._distributions):
                assert_almost_equal(d.samples, b._distributions[i].samples)
            for i, d in enumerate(a.euclidian_displacements):
                assert_equal(d.samples, b.euclidian_displacements[i].samples)
            assert a.dt.size == b.dt.size
            assert isinstance(b.dt, np.ndarray)
            assert_equal(a._diffusion_coefficient.samples, b._diffusion_coefficient.samples)
            assert_equal(a.intercept.samples, b.intercept.samples)

    def test_bootstrap(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion()
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_bootstrap_dt_skip(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o)
            bs.diffusion(dt_skip=150)
            assert bs.covariance_matrix.shape == (9, 9)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_bootstrap_use_ngp(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(200, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(use_ngp=True)
            assert bs.covariance_matrix.shape == (190 - np.argmax(bs.ngp), 190 - np.argmax(bs.ngp))
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_bootstrap_fit_intercept(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(n_samples=500, fit_intercept=False)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert bs._diffusion_coefficient.size == 1600
            assert bs.intercept is None

    def test_bootstrap_n_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(n_samples=100)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 320
            assert bs.intercept.size == 320

    def test_bootstrap_D(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(n_samples=100)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.D.size == 320
            assert bs.intercept.size == 320

    def test_bootstrap_thin(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(use_ngp=True, thin=1)
            assert bs.covariance_matrix.shape == (10 - np.argmax(bs.ngp), 10 - np.argmax(bs.ngp))
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 32000
            assert bs.intercept.size == 32000

    # Waiting on https://github.com/dfm/emcee/pull/376
    # def test_initialisation_random_state(self):
    #     disp_3d = [RNG.randn(100, i, 3) + 1000 for i in range(20, 10, -1)]
    #     dt = np.linspace(100, 1000, 10)
    #     bs1 = DiffBootstrap(dt, disp_3d, n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert bs1.covariance_matrix.shape == (10, 10)
    #     assert isinstance(bs1.diffusion_coefficient, Distribution)
    #     assert isinstance(bs1.intercept, Distribution)
    #     assert bs1.diffusion_coefficient.size == 500
    #     assert bs1.intercept.size == 500
    #     bs2 = DiffBootstrap(dt, disp_3d, n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert_almost_equal(bs1.v, bs2.v)
    #     assert_almost_equal(bs1.covariance_matrix, bs2.covariance_matrix)
    #     assert_almost_equal(bs1.diffusion_coefficient.samples, bs2.diffusion_coefficient.samples)


class TestMSTDBootstrap(unittest.TestCase):
    """
    Tests for the diffusion.MSTDBootstrap class.
    """

    def test_initialisation(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000

    def test_initialisation_n_resamples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, n_resamples=10, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 10

    def test_initialisation_max_resamples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt,
                               disp_3d,
                               n_o,
                               n_resamples=10,
                               max_resamples=100,
                               random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size <= 110

    def test_initialisation_random_state(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs1 = MSTDBootstrap(dt, disp_3d, n_o, bootstrap=True, random_state=np.random.RandomState(0))
            assert bs1.n.shape == (5, )
            assert bs1.s.shape == (5, )
            assert_almost_equal(bs1.v, np.square(bs1.s))
            assert bs1.ngp.shape == (5, )
            assert len(bs1.euclidian_displacements) == 5
            for i in bs1.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs1._distributions:
                assert i.samples.size >= 1000
            bs2 = MSTDBootstrap(dt, disp_3d, n_o, bootstrap=True, random_state=np.random.RandomState(0))
            assert bs1._distributions[-1].size == bs2._distributions[-1].size
            assert_almost_equal(bs1._distributions[-1].samples, bs2._distributions[-1].samples)

    def test_initialisation_progress(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000
            assert isinstance(bs._iterator, range)

    def test_initialisation_skip_where_low_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(1, i, 3) for i in range(10, 1, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (4, )
            assert bs.s.shape == (4, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (4, )
            assert len(bs.euclidian_displacements) == 4
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000
            assert isinstance(bs._iterator, range)

    def test_bootstrap(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion()
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_bootstrap_use_ngp(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(use_ngp=True)
            assert bs.covariance_matrix.shape == (5 - np.argmax(bs.ngp), 5 - np.argmax(bs.ngp))
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_bootstrap_fit_intercept(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(n_samples=500, fit_intercept=False)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert bs._jump_diffusion_coefficient.size == 1600
            assert bs.intercept is None

    def test_bootstrap_n_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 320
            assert bs.intercept.size == 320

    def test_bootstrap_D(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 320
            assert bs.intercept.size == 320

    def test_bootstrap_thin(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(200, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSTDBootstrap(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(use_ngp=True, thin=1)
            assert bs.covariance_matrix.shape == (95 - np.argmax(bs.ngp), 95 - np.argmax(bs.ngp))
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 32000
            assert bs.intercept.size == 32000

    # Waiting on https://github.com/dfm/emcee/pull/376
    # def test_initialisation_random_state(self):
    #     disp_3d = [RNG.randn(100, i, 3) + 1000 for i in range(20, 10, -1)]
    #     dt = np.linspace(100, 1000, 10)
    #     bs1 = MSTDBootstrap(dt, disp_3d, random_state=np.random.RandomState(0))
    #     bs1.jump_diffusion( n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert bs1.covariance_matrix.shape == (10, 10)
    #     assert isinstance(bs1.diffusion_coefficient, Distribution)
    #     assert isinstance(bs1.intercept, Distribution)
    #     assert bs1.diffusion_coefficient.size == 500
    #     assert bs1.intercept.size == 500
    #     bs2 = MSTDBootstrap(dt, disp_3d, random_state=np.random.RandomState(0))
    #     bs2.jump_diffusion( n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert_almost_equal(bs1.v, bs2.v)
    #     assert_almost_equal(bs1.covariance_matrix, bs2.covariance_matrix)
    #     assert_almost_equal(bs1.diffusion_coefficient.samples, bs2.diffusion_coefficient.samples)


class TestMSCDBootstrap(unittest.TestCase):
    """
    Tests for the diffusion.MSCDBootstrap class.
    """

    def test_initialisation(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000

    def test_initialisation_n_resamples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, np.ones(100), n_o, n_resamples=10, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 10

    def test_initialisation_max_resamples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt,
                               disp_3d,
                               1,
                               n_o,
                               n_resamples=10,
                               max_resamples=100,
                               random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size <= 110

    def test_initialisation_random_state(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs1 = MSCDBootstrap(dt, disp_3d, 1, n_o, bootstrap=True, random_state=np.random.RandomState(0))
            assert bs1.n.shape == (5, )
            assert bs1.s.shape == (5, )
            assert_almost_equal(bs1.v, np.square(bs1.s))
            assert bs1.ngp.shape == (5, )
            assert len(bs1.euclidian_displacements) == 5
            for i in bs1.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs1._distributions:
                assert i.samples.size >= 1000
            bs2 = MSCDBootstrap(dt, disp_3d, 1, n_o, bootstrap=True, random_state=np.random.RandomState(0))
            assert bs1._distributions[-1].size == bs2._distributions[-1].size
            assert_almost_equal(bs1._distributions[-1].samples, bs2._distributions[-1].samples)

    def test_initialisation_progress(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000
            assert isinstance(bs._iterator, range)

    def test_initialisation_skip_where_low_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(1, i, 3) for i in range(10, 1, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (4, )
            assert bs.s.shape == (4, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (4, )
            assert len(bs.euclidian_displacements) == 4
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs._distributions:
                assert i.samples.size >= 1000
            assert isinstance(bs._iterator, range)

    def test_bootstrap(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(1, 10)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 3200
            assert bs.intercept.size == 3200

    def test_bootstrap_use_ngp(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(200, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(1, 10, use_ngp=True)
            assert bs.covariance_matrix.shape == (95 - np.argmax(bs.ngp), 95 - np.argmax(bs.ngp))
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 3200
            assert bs.intercept.size == 3200

    def test_bootstrap_fit_intercept(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(1, 10, n_samples=500, fit_intercept=False)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert bs.sigma.size == 1600
            assert bs.intercept is None

    def test_bootstrap_n_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(1, 10, n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 320
            assert bs.intercept.size == 320

    def test_bootstrap_D(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(1, 10, n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 320
            assert bs.intercept.size == 320

    def test_bootstrap_thin(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(200, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSCDBootstrap(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(1, 10, use_ngp=True, thin=1)
            assert bs.covariance_matrix.shape == (95 - np.argmax(bs.ngp), 95 - np.argmax(bs.ngp))
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 32000
            assert bs.intercept.size == 32000

    # Waiting on https://github.com/dfm/emcee/pull/376
    # def test_initialisation_random_state(self):
    #     disp_3d = [RNG.randn(100, i, 3) + 1000 for i in range(20, 10, -1)]
    #     dt = np.linspace(100, 1000, 10)
    #     bs1 = DiffBootstrap(dt, disp_3d, n_samples=100, random_state=np.random.RandomState(0))
    #     assert bs1.covariance_matrix.shape == (10, 10)
    #     assert isinstance(bs1.diffusion_coefficient, Distribution)
    #     assert isinstance(bs1.intercept, Distribution)
    #     assert bs1.diffusion_coefficient.size == 500
    #     assert bs1.intercept.size == 500
    #     bs2 = DiffBootstrap(dt, disp_3d, n_samples=100, random_state=np.random.RandomState(0))
    #     assert_almost_equal(bs1.v, bs2.v)
    #     assert_almost_equal(bs1.covariance_matrix, bs2.covariance_matrix)
    #     assert_almost_equal(bs1.diffusion_coefficient.samples, bs2.diffusion_coefficient.samples)


class TestFunctions(unittest.TestCase):
    """
    Testing other functions.
    """

    def test__bootstrap_random(self):
        result1 = _bootstrap(np.arange(1, 100, 1), 200, 100, np.random.RandomState(0))
        result2 = _bootstrap(np.arange(1, 100, 1), 200, 100, np.random.RandomState(0))
        assert_almost_equal(result1, result2)
