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
from kinisi.diffusion import Bootstrap, MSDBootstrap, _bootstrap
from uravu.distribution import Distribution


RNG = np.random.RandomState(42)


class TestBootstrap(unittest.TestCase):
    """
    Tests for the diffusion.Bootstrap class.
    """
    def test_initialisation(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        dt = np.linspace(100, 1000, 10)
        bs = Bootstrap(dt, disp_3d)
        for i, d in enumerate(disp_3d):
            assert_almost_equal(bs.displacements[i], d)
        assert_almost_equal(bs.delta_t, dt)
        assert bs.max_obs == 20
        assert bs.distributions == []
        assert bs.dt.size == 0
        assert isinstance(bs.dt, np.ndarray)
        assert isinstance(bs.iterator, tqdm)
        assert bs.diffusion_coefficient == None
        assert bs.intercept == None

    def test_initialisation_sub_sample_dt(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        dt = np.linspace(100, 1000, 10)
        bs = Bootstrap(dt, disp_3d, sub_sample_dt=2)
        for i, d in enumerate(disp_3d[::2]):
            assert_almost_equal(bs.displacements[i], d)
        assert_almost_equal(bs.delta_t, dt[::2])
        assert bs.max_obs == 20
        assert bs.distributions == []
        assert bs.dt.size == 0
        assert isinstance(bs.dt, np.ndarray)
        assert isinstance(bs.iterator, tqdm)
        assert bs.diffusion_coefficient == None
        assert bs.intercept == None

    def test_initialisation_progress(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        dt = np.linspace(100, 1000, 10)
        bs = Bootstrap(dt, disp_3d, progress=False)
        for i, d in enumerate(disp_3d):
            assert_almost_equal(bs.displacements[i], d)
        assert_almost_equal(bs.delta_t, dt)
        assert bs.max_obs == 20
        assert bs.distributions == []
        assert bs.dt.size == 0
        assert isinstance(bs.dt, np.ndarray)
        assert isinstance(bs.iterator, range)
        assert bs.diffusion_coefficient == None
        assert bs.intercept == None

    def test_iterator_true(self):
        result = Bootstrap.iterator(True, range(10))
        assert isinstance(result, tqdm)

    def test_iterator_false(self):
        result = Bootstrap.iterator(False, range(10))
        assert isinstance(result, range)


class TestMSDBootstrap(unittest.TestCase):
    """
    Tests for the diffusion.MSDBootstrap class.
    """
    def test_initialisation(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=np.random.RandomState(0))
            assert bs.msd.shape == (10,)
            assert bs.msd_std.shape == (10,)
            assert_almost_equal(bs.msd_var, np.square(bs.msd_std))
            assert_equal(bs.n_samples_msd, [2000, 1000, 666, 500, 400, 333, 285, 250, 222, 200])
            assert bs.ngp.shape == (10,)
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs.distributions:
                assert i.samples.size >= 1000
                assert_almost_equal(i.ci_points, [2.5, 97.5])
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)
    
    def test_initialisation_n_resamples(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_resamples=10, random_state=np.random.RandomState(0))
            assert bs.msd.shape == (10,)
            assert bs.msd_std.shape == (10,)
            assert_almost_equal(bs.msd_var, np.square(bs.msd_std))
            assert_equal(bs.n_samples_msd, [2000, 1000, 666, 500, 400, 333, 285, 250, 222, 200])
            assert bs.ngp.shape == (10,)
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs.distributions:
                assert i.samples.size >= 10
                assert_almost_equal(i.ci_points, [2.5, 97.5])
            assert issubclass(w[0].category, UserWarning)
            assert "kurtosistest" in str(w[0].message)

    def test_initialisation_max_resamples(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, n_resamples=10, max_resamples=100, random_state=np.random.RandomState(0))
            assert bs.msd.shape == (10,)
            assert bs.msd_std.shape == (10,)
            assert_almost_equal(bs.msd_var, np.square(bs.msd_std))
            assert_equal(bs.n_samples_msd, [2000, 1000, 666, 500, 400, 333, 285, 250, 222, 200])
            assert bs.ngp.shape == (10,)
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs.distributions:
                assert i.samples.size <= 110
                assert_almost_equal(i.ci_points, [2.5, 97.5])
            assert issubclass(w[0].category, UserWarning)
            assert "kurtosistest" in str(w[0].message)
        
    def test_initialisation_confidence_interval(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        dt = np.linspace(100, 1000, 10)
        bs = MSDBootstrap(dt, disp_3d, confidence_interval=[25, 75], random_state=np.random.RandomState(0))
        assert bs.msd.shape == (10,)
        assert bs.msd_std.shape == (10,)
        assert_almost_equal(bs.msd_var, np.square(bs.msd_std))
        assert_equal(bs.n_samples_msd, [2000, 1000, 666, 500, 400, 333, 285, 250, 222, 200])
        assert bs.ngp.shape == (10,)
        assert len(bs.euclidian_displacements) == 10
        for i in bs.euclidian_displacements:
            assert isinstance(i, Distribution)
        for i in bs.distributions:
            assert i.samples.size >= 1000
            assert_almost_equal(i.ci_points, [25, 75])

    def test_initialisation_random_state(self):
        disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
        dt = np.linspace(100, 1000, 10)
        bs1 = MSDBootstrap(dt, disp_3d, random_state=np.random.RandomState(0))
        assert bs1.msd.shape == (10,)
        assert bs1.msd_std.shape == (10,)
        assert_almost_equal(bs1.msd_var, np.square(bs1.msd_std))
        assert_equal(bs1.n_samples_msd, [2000, 1000, 666, 500, 400, 333, 285, 250, 222, 200])
        assert bs1.ngp.shape == (10,)
        assert len(bs1.euclidian_displacements) == 10
        for i in bs1.euclidian_displacements:
            assert isinstance(i, Distribution)
        for i in bs1.distributions:
            # assert i.samples.size >= 1000
            assert_almost_equal(i.ci_points, [2.5, 97.5])
        bs2 = MSDBootstrap(dt, disp_3d, random_state=np.random.RandomState(0))
        assert bs1.distributions[-1].size == bs2.distributions[-1].size
        assert_almost_equal(bs1.distributions[-1].samples, bs2.distributions[-1].samples)

    def test_initialisation_progress(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, progress=False, random_state=np.random.RandomState(0))
            assert bs.msd.shape == (10,)
            assert bs.msd_std.shape == (10,)
            assert_almost_equal(bs.msd_var, np.square(bs.msd_std))
            assert_equal(bs.n_samples_msd, [2000, 1000, 666, 500, 400, 333, 285, 250, 222, 200])
            assert bs.ngp.shape == (10,)
            assert len(bs.euclidian_displacements) == 10
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs.distributions:
                assert i.samples.size >= 1000
                assert_almost_equal(i.ci_points, [2.5, 97.5])
            assert isinstance(bs.iterator, range)
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)

    def test_initialisation_skip_where_low_samples(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(1, i, 3) for i in range(10, 1, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, progress=False, random_state=np.random.RandomState(0))
            assert bs.msd.shape == (5,)
            assert bs.msd_std.shape == (5,)
            assert_almost_equal(bs.msd_var, np.square(bs.msd_std))
            assert_equal(bs.n_samples_msd, [10, 5, 3, 2, 2])
            assert bs.ngp.shape == (5,)
            assert len(bs.euclidian_displacements) == 5
            for i in bs.euclidian_displacements:
                assert isinstance(i, Distribution)
            for i in bs.distributions:
                assert i.samples.size >= 1000
                assert_almost_equal(i.ci_points, [2.5, 97.5])
            assert isinstance(bs.iterator, range)
            assert issubclass(w[0].category, UserWarning)
            assert "kurtosistest" in str(w[0].message)

    def test_n_samples_a(self):
        result = MSDBootstrap.n_samples((100, 10), 10)
        assert result == 1000

    def test_n_samples_b(self):
        result = MSDBootstrap.n_samples((100, 8), 10)
        assert result == 333

    def test_sample_until_normal(self):
        distro1 = MSDBootstrap.sample_until_normal(RNG.randn(1000), 5, 100, 10000)
        assert distro1.size == 100
 
    def test_sample_until_normal_random(self):
        with warnings.catch_warnings(record=True) as w:
            distro1 = MSDBootstrap.sample_until_normal(np.arange(1, 10, 1), 5, 100, 100, random_state=np.random.RandomState(0))
            distro2 = MSDBootstrap.sample_until_normal(np.arange(1, 10, 1), 5, 100, 100, random_state=np.random.RandomState(0))
            assert_almost_equal(distro1.samples, distro2.samples)
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)

    def test_ngp_calculation(self):
        result = MSDBootstrap.ngp_calculation(np.array([1, 2, 3]))
        assert_almost_equal(result, -0.3)

    def test_bootstrap(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=RNG)
            bs.diffusion()
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs.diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.diffusion_coefficient.size == 32000
            assert bs.intercept.size == 32000
            assert issubclass(w[0].category, UserWarning)
            assert "covariance" in str(w[0].message)

    def test_bootstrap_use_ngp(self):
        with warnings.catch_warnings(record=True) as w: 
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=RNG)
            bs.diffusion(use_ngp=True)
            assert bs.covariance_matrix.shape == (10-np.argmax(bs.ngp), 10-np.argmax(bs.ngp))
            assert isinstance(bs.diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.diffusion_coefficient.size == 32000
            assert bs.intercept.size == 32000

    def test_bootstrap_fit_intercept(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=RNG)
            bs.diffusion(n_walkers=5, n_samples=100, fit_intercept=False)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs.diffusion_coefficient, Distribution)
            assert bs.diffusion_coefficient.size == 500
            assert bs.intercept == None
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)

    def test_bootstrap_n_walkers(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=RNG)
            bs.diffusion(n_walkers=5)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs.diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.diffusion_coefficient.size == 5000
            assert bs.intercept.size == 5000
            assert issubclass(w[0].category, UserWarning)
            assert "covariance" in str(w[0].message)

    def test_bootstrap_n_samples(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=RNG)
            bs.diffusion(n_samples=100)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs.diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.diffusion_coefficient.size == 3200
            assert bs.intercept.size == 3200
            assert issubclass(w[0].category, UserWarning)
            assert "covariance" in str(w[0].message)
    
    def test_bootstrap_D(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=RNG)
            bs.diffusion(n_samples=100)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs.diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.D.size == 3200
            assert bs.intercept.size == 3200
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)

    def test_bootstrap_D_offset(self):
        with warnings.catch_warnings(record=True) as w:
            disp_3d = [RNG.randn(100, i, 3) for i in range(20, 10, -1)]
            dt = np.linspace(100, 1000, 10)
            bs = MSDBootstrap(dt, disp_3d, random_state=RNG)
            bs.diffusion(n_samples=100)
            assert bs.covariance_matrix.shape == (10, 10)
            assert isinstance(bs.diffusion_coefficient, Distribution)
            assert isinstance(bs.intercept, Distribution)
            assert bs.diffusion_coefficient.size == 3200
            assert bs.D_offset.size == 3200
            assert issubclass(w[0].category, UserWarning)
            assert "covariance" in str(w[0].message)  

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
    #     assert_almost_equal(bs1.msd_var, bs2.msd_var)
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