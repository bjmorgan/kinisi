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
from kinisi.diffusion import Diffusion, MSDDiffusion, MSTDDiffusion, MSCDDiffusion
from uravu.distribution import Distribution

RNG = np.random.RandomState(43)


class TestDiffusion(unittest.TestCase):
    """
    Tests for the diffusion.Diffusion class.
    """

    def test_dictionary_roundtrip(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        n_o = np.ones(len(disp_3d)) * 100
        dt = np.linspace(100, 1000, 10)
        a = Diffusion(dt, disp_3d, n_o)
        b = Diffusion.from_dict(a.to_dict())
        for i, d in enumerate(disp_3d):
            assert_almost_equal(a._displacements[i], b._displacements[i])
        assert_almost_equal(a._delta_t, b._delta_t)
        assert a._max_obs == b._max_obs
        assert a.dt.size == b.dt.size
        assert isinstance(b.dt, np.ndarray)
        assert b._diffusion_coefficient is None
        assert b.intercept is None

    def test_initialisation(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        n_o = np.ones(len(disp_3d)) * 100
        dt = np.linspace(100, 1000, 10)
        bs = Diffusion(dt, disp_3d, n_o)
        for i, d in enumerate(disp_3d):
            assert_almost_equal(bs._displacements[i], d)
        assert_almost_equal(bs._delta_t, dt)
        assert bs._max_obs == 20
        assert bs.dt.size == 0
        assert isinstance(bs.dt, np.ndarray)
        assert bs._diffusion_coefficient is None
        assert bs.intercept is None

    def test_initialisation_sub_sample_dt(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        n_o = np.ones(len(disp_3d)) * 100
        dt = np.linspace(100, 1000, 10)
        bs = Diffusion(dt, disp_3d, n_o, sub_sample_dt=2)
        for i, d in enumerate(disp_3d[::2]):
            assert_almost_equal(bs._displacements[i], d)
        assert_almost_equal(bs._delta_t, dt[::2])
        assert bs._max_obs == 20
        assert bs.dt.size == 0
        assert isinstance(bs.dt, np.ndarray)
        assert bs._diffusion_coefficient is None
        assert bs.intercept is None

    def test_iterator_true(self):
        result = Diffusion.iterator(True, range(10))
        assert isinstance(result, tqdm)

    def test_iterator_false(self):
        result = Diffusion.iterator(False, range(10))
        assert isinstance(result, range)

    def test_ngp_calculation(self):
        result = Diffusion.ngp_calculation(np.array([1, 2, 3]))
        assert_almost_equal(result, -0.3)


class TestMSDDiffusion(unittest.TestCase):
    """
    Tests for the diffusion.MSDDiffusion class.
    """

    def test_dictionary_roundtrip(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            a = MSDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            b = MSDDiffusion.from_dict(a.to_dict())
            for i, d in enumerate(disp_3d):
                assert_almost_equal(a._displacements[i], b._displacements[i])
            assert_almost_equal(a._delta_t, b._delta_t)
            assert a._max_obs == b._max_obs
            assert_equal(a.n, b.n)
            assert_equal(a.s, b.s)
            assert_equal(a.v, b.v)
            assert_equal(a.ngp, b.ngp)
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
            bs = MSDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            assert bs.n.shape == (10, )
            assert bs.s.shape == (10, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (10, )
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)

    def test_initialisation_random_state(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs1 = MSDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            assert bs1.n.shape == (10, )
            assert bs1.s.shape == (10, )
            assert_almost_equal(bs1.v, np.square(bs1.s))
            assert bs1.ngp.shape == (10, )
            assert len(bs1.euclidian_displacements) == 10
            for i in bs1.euclidian_displacements:
                assert isinstance(i, Distribution)
            bs2 = MSDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))

    def test_initialisation_progress(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (10, )
            assert bs.s.shape == (10, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (10, )
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            assert isinstance(bs._iterator, range)

    def test_initialisation_skip_where_low_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(1, i, 3) for i in range(10, 1, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (9, )
            assert bs.s.shape == (9, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (9, )
            assert len(bs.euclidian_displacements) == 9
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            assert isinstance(bs._iterator, range)

    def test_diffusion_dictionary_roundtrip(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            a = MSDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            a.diffusion(0)
            b = MSDDiffusion.from_dict(a.to_dict())
            for i, d in enumerate(disp_3d):
                assert_almost_equal(a._displacements[i], b._displacements[i])
            assert_almost_equal(a._delta_t, b._delta_t)
            assert a._max_obs == b._max_obs
            assert_equal(a.n, b.n)
            assert_equal(a.s, b.s)
            assert_equal(a.v, b.v)
            assert_equal(a.ngp, b.ngp)
            for i, d in enumerate(a.euclidian_displacements):
                assert_equal(d.samples, b.euclidian_displacements[i].samples)
            assert a.dt.size == b.dt.size
            assert isinstance(b.dt, np.ndarray)
            assert_equal(a._diffusion_coefficient.samples, b._diffusion_coefficient.samples)
            assert_equal(a.intercept.samples, b.intercept.samples)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(0)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_diffusion_dt_skip(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o)
            bs.diffusion(150)
            assert bs.covariance_matrix.shape == (9, 9)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_diffusion_use_ngp(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(200, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(dt[bs.ngp.argmax()])
            assert bs.covariance_matrix.shape == (190 - np.argmax(bs.ngp), 190 - np.argmax(bs.ngp))
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_diffusion_fit_intercept(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(0, n_samples=500, fit_intercept=False)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert bs._diffusion_coefficient.size == 1600
            assert bs.intercept is None

    def test_diffusion_n_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(0, n_samples=100)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 320
            assert bs.intercept.size == 320

    def test_diffusion_D(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(0, n_samples=100)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.D.size == 320
            assert bs.intercept.size == 320

    def test_diffusion_thin(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(dt[bs.ngp.argmax()], thin=1)
            assert bs.covariance_matrix.shape == (10 - np.argmax(bs.ngp), 10 - np.argmax(bs.ngp))
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 32000
            assert bs.intercept.size == 32000

    def test_diffusion_ppd(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.diffusion(0)
            ppd = bs.posterior_predictive(128, 128)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200
            assert ppd.shape == (128 * 128, 5)

    # Waiting on https://github.com/dfm/emcee/pull/376
    # def test_initialisation_random_state(self):
    #     disp_3d = [RNG.randn(100, i, 3) + 1000 for i in range(20, 10, -1)]
    #     dt = np.linspace(100, 1000, 10)
    #     bs1 = DiffDiffusion(dt, disp_3d, n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert bs1.covariance_matrix.shape == (10, 10)
    #     assert isinstance(bs1.diffusion_coefficient, Distribution)
    #     assert isinstance(bs1.intercept, Distribution)
    #     assert bs1.diffusion_coefficient.size == 500
    #     assert bs1.intercept.size == 500
    #     bs2 = DiffDiffusion(dt, disp_3d, n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert_almost_equal(bs1.v, bs2.v)
    #     assert_almost_equal(bs1.covariance_matrix, bs2.covariance_matrix)
    #     assert_almost_equal(bs1.diffusion_coefficient.samples, bs2.diffusion_coefficient.samples)


class TestMSTDDiffusion(unittest.TestCase):
    """
    Tests for the diffusion.MSTDDiffusion class.
    """

    def test_initialisation(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)

    def test_initialisation_random_state(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs1 = MSTDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))
            assert bs1.n.shape == (5, )
            assert bs1.s.shape == (5, )
            assert_almost_equal(bs1.v, np.square(bs1.s))
            assert bs1.ngp.shape == (5, )
            assert len(bs1.euclidian_displacements) == 5
            for i in bs1.euclidian_displacements:
                assert isinstance(i, Distribution)
            bs2 = MSTDDiffusion(dt, disp_3d, n_o, random_state=np.random.RandomState(0))

    def test_initialisation_progress(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            assert isinstance(bs._iterator, range)

    def test_initialisation_skip_where_low_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(1, i, 3) for i in range(10, 1, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (4, )
            assert bs.s.shape == (4, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (4, )
            assert len(bs.euclidian_displacements) == 4
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            assert isinstance(bs._iterator, range)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(0)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_diffusion_use_ngp(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(dt[bs.ngp.argmax()])
            assert bs.covariance_matrix.shape == (5 - np.argmax(bs.ngp), 5 - np.argmax(bs.ngp))
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200

    def test_diffusion_fit_intercept(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(0, n_samples=500, fit_intercept=False)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert bs._jump_diffusion_coefficient.size == 1600
            assert bs.intercept is None

    def test_diffusion_n_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(0, n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 320
            assert bs.intercept.size == 320

    def test_diffusion_D(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(0, n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 320
            assert bs.intercept.size == 320

    def test_diffusion_thin(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(200, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(dt[bs.ngp.argmax()], thin=1)
            assert bs.covariance_matrix.shape == (95 - np.argmax(bs.ngp), 95 - np.argmax(bs.ngp))
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 32000
            assert bs.intercept.size == 32000

    def test_diffusion_ppd(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.jump_diffusion(0)
            ppd = bs.posterior_predictive(128, 128)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs._jump_diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs._jump_diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200
            assert ppd.shape == (128 * 128, 5)

    # Waiting on https://github.com/dfm/emcee/pull/376
    # def test_initialisation_random_state(self):
    #     disp_3d = [RNG.randn(100, i, 3) + 1000 for i in range(20, 10, -1)]
    #     dt = np.linspace(100, 1000, 10)
    #     bs1 = MSTDDiffusion(dt, disp_3d, random_state=np.random.RandomState(0))
    #     bs1.jump_diffusion( n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert bs1.covariance_matrix.shape == (10, 10)
    #     assert isinstance(bs1.diffusion_coefficient, Distribution)
    #     assert isinstance(bs1.intercept, Distribution)
    #     assert bs1.diffusion_coefficient.size == 500
    #     assert bs1.intercept.size == 500
    #     bs2 = MSTDDiffusion(dt, disp_3d, random_state=np.random.RandomState(0))
    #     bs2.jump_diffusion( n_walkers=5, n_samples=100, random_state=np.random.RandomState(0))
    #     assert_almost_equal(bs1.v, bs2.v)
    #     assert_almost_equal(bs1.covariance_matrix, bs2.covariance_matrix)
    #     assert_almost_equal(bs1.diffusion_coefficient.samples, bs2.diffusion_coefficient.samples)


class TestMSCDDiffusion(unittest.TestCase):
    """
    Tests for the diffusion.MSCDDiffusion class.
    """

    def test_initialisation(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)

    def test_initialisation_random_state(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs1 = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=np.random.RandomState(0))
            assert bs1.n.shape == (5, )
            assert bs1.s.shape == (5, )
            assert_almost_equal(bs1.v, np.square(bs1.s))
            assert bs1.ngp.shape == (5, )
            assert len(bs1.euclidian_displacements) == 5
            for i in bs1.euclidian_displacements:
                assert isinstance(i, Distribution)
            bs2 = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=np.random.RandomState(0))

    def test_initialisation_progress(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (5, )
            assert bs.s.shape == (5, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (5, )
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            assert isinstance(bs._iterator, range)

    def test_initialisation_skip_where_low_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(1, i, 3) for i in range(10, 1, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, progress=False, random_state=np.random.RandomState(0))
            assert bs.n.shape == (4, )
            assert bs.s.shape == (4, )
            assert_almost_equal(bs.v, np.square(bs.s))
            assert bs.ngp.shape == (4, )
            assert len(bs.euclidian_displacements) == 4
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            assert isinstance(bs._iterator, range)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(0, 1, 10)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 3200
            assert bs.intercept.size == 3200

    def test_diffusion_use_ngp(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(200, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(dt[bs.ngp.argmax()], 1, 10)
            assert bs.covariance_matrix.shape == (95 - np.argmax(bs.ngp), 95 - np.argmax(bs.ngp))
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 3200
            assert bs.intercept.size == 3200

    def test_diffusion_fit_intercept(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(0, 1, 10, n_samples=500, fit_intercept=False)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert bs.sigma.size == 1600
            assert bs.intercept is None

    def test_diffusion_n_samples(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(0, 1, 10, n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 320
            assert bs.intercept.size == 320

    def test_diffusion_D(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(0, 1, 10, n_samples=100)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 320
            assert bs.intercept.size == 320

    def test_diffusion_thin(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(210, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 190)
            bs = MSCDDiffusion(dt, disp_3d, 1, n_o, random_state=RNG)
            bs.conductivity(dt[bs.ngp.argmax()], 1, 10, thin=1)
            assert bs.covariance_matrix.shape == (100 - np.argmax(bs.ngp), 100 - np.argmax(bs.ngp))
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 32000
            assert bs.intercept.size == 32000

    def test_diffusion_ppd(self):
        with warnings.catch_warnings(record=True) as _:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            n_o = np.ones(len(disp_3d)) * 100
            dt = np.linspace(100, 1000, 10)
            bs = MSTDDiffusion(dt, disp_3d, n_o, random_state=RNG)
            bs.conductivity(0, 1, 10)
            ppd = bs.posterior_predictive(128, 128)
            assert bs.covariance_matrix.shape == (5, 5)
            assert isinstance(bs.sigma, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.sigma.size == 3200
            assert bs.intercept.size == 3200
            assert ppd.shape == (128 * 128, 5)

    # Waiting on https://github.com/dfm/emcee/pull/376
    # def test_initialisation_random_state(self):
    #     disp_3d = [RNG.randn(100, i, 3) + 1000 for i in range(20, 10, -1)]
    #     dt = np.linspace(100, 1000, 10)
    #     bs1 = DiffDiffusion(dt, disp_3d, n_samples=100, random_state=np.random.RandomState(0))
    #     assert bs1.covariance_matrix.shape == (10, 10)
    #     assert isinstance(bs1.diffusion_coefficient, Distribution)
    #     assert isinstance(bs1.intercept, Distribution)
    #     assert bs1.diffusion_coefficient.size == 500
    #     assert bs1.intercept.size == 500
    #     bs2 = DiffDiffusion(dt, disp_3d, n_samples=100, random_state=np.random.RandomState(0))
    #     assert_almost_equal(bs1.v, bs2.v)
    #     assert_almost_equal(bs1.covariance_matrix, bs2.covariance_matrix)
    #     assert_almost_equal(bs1.diffusion_coefficient.samples, bs2.diffusion_coefficient.samples)
