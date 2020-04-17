"""
Tests for diffusion module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
import uncertainties
from numpy.testing import assert_equal, assert_almost_equal
from kinisi import diffusion, UREG
from uravu.distribution import Distribution
from uravu.utils import straight_line


class TestMsd(unittest.TestCase):
    """
    Unit tests for diffusion module
    """

    def test_msd_bootstrap_a(self):
        """
        Test msd_bootstrap for initial normal.
        """
        ordinate = np.random.randn(100, 50, 3)
        to_resample = [
            np.array(ordinate) for i in range(1, 6)
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.msd_bootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=100,
            max_resamples=1000)
        assert_equal(delta_t.size, 5)
        assert_equal(mean.size, 5)
        assert_equal(err.size, 5)
        assert_equal(con_int_l.size, 5)
        assert_equal(con_int_u.size, 5)

    def test_msd_bootstrap_b(self):
        """
        Test msd_bootstrap for initial uniform with progress.
        """
        ordinate = np.random.uniform(size=(100, 50, 3))
        to_resample = [
            np.array(ordinate) for i in range(1, 6)
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.msd_bootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=True,
            n_resamples=1,
            max_resamples=1000)
        assert_equal(delta_t.size, 5)
        assert_equal(mean.size, 5)
        assert_equal(err.size, 5)
        assert_equal(con_int_l.size, 5)
        assert_equal(con_int_u.size, 5)

    def test_msd_bootstrap_c(self):
        """
        Test msd_bootstrap for to go over the limit point.
        """
        ordinate1 = np.random.randn(5, 500, 3)
        ordinate2 = np.random.randn(5, 450, 3)
        ordinate3 = np.random.randn(5, 400, 3)
        ordinate4 = np.random.randn(5, 100, 3)
        ordinate5 = np.random.randn(5, 10, 3)
        to_resample = [
            ordinate1,
            ordinate2,
            ordinate3,
            ordinate4,
            ordinate5,
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.msd_bootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=1,
            max_resamples=10, 
            samples_freq=2)
        assert_equal(delta_t.size, 3)
        assert_equal(mean.size, 3)
        assert_equal(err.size, 3)
        assert_equal(con_int_l.size, 3)
        assert_equal(con_int_u.size, 3)
    
    def test_msd_bootstrap_d(self):
        """
        Test msd_bootstrap very few particles.
        """
        ordinate1 = np.random.randn(10, 10, 3)
        ordinate2 = np.random.randn(1, 1, 3)
        ordinate3 = np.random.randn(1, 1, 3)
        ordinate4 = np.random.randn(1, 1, 3)
        ordinate5 = np.random.randn(1, 1, 3)
        to_resample = [
            ordinate1,
            ordinate2,
            ordinate3,
            ordinate4,
            ordinate5,
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.msd_bootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=1,
            max_resamples=10, 
            samples_freq=2)
        assert_equal(delta_t.size, 1)
        assert_equal(mean.size, 1)
        assert_equal(err.size, 1)
        assert_equal(con_int_l.size, 1)
        assert_equal(con_int_u.size, 1)

    def test_mscd_bootstrap_a(self):
        """
        Test mscd_bootstrap for initial normal.
        """
        ordinate = np.random.randn(100, 50, 3)
        to_resample = [
            np.array(ordinate) for i in range(1, 6)
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.mscd_bootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            [1],
            progress=False,
            n_resamples=100,
            max_resamples=1000)
        assert_equal(delta_t.size, 5)
        assert_equal(mean.size, 5)
        assert_equal(err.size, 5)
        assert_equal(con_int_l.size, 5)
        assert_equal(con_int_u.size, 5)

    def test_mscd_bootstrap_b(self):
        """
        Test mscd_bootstrap for initial uniform with progress.
        """
        ordinate = np.random.uniform(size=(100, 50, 3))
        to_resample = [
            np.array(ordinate) for i in range(1, 6)
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.mscd_bootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            [1],
            progress=True,
            n_resamples=1,
            max_resamples=1000)
        assert_equal(delta_t.size, 5)
        assert_equal(mean.size, 5)
        assert_equal(err.size, 5)
        assert_equal(con_int_l.size, 5)
        assert_equal(con_int_u.size, 5)

    def test_mscd_bootstrap_c(self):
        """
        Test mscd_bootstrap for to go over the limit point.
        """
        ordinate1 = np.random.randn(5, 500, 3)
        ordinate2 = np.random.randn(5, 450, 3)
        ordinate3 = np.random.randn(5, 400, 3)
        ordinate4 = np.random.randn(5, 100, 3)
        ordinate5 = np.random.randn(5, 10, 3)
        to_resample = [
            ordinate1,
            ordinate2,
            ordinate3,
            ordinate4,
            ordinate5,
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.mscd_bootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            [1],
            progress=False,
            n_resamples=1,
            max_resamples=10, 
            samples_freq=2)
        assert_equal(delta_t.size, 2)
        assert_equal(mean.size, 2)
        assert_equal(err.size, 2)
        assert_equal(con_int_l.size, 2)
        assert_equal(con_int_u.size, 2)

    def test_diffusion_init(self):
        """
        Test the initialisation of diffusion
        """
        dt = np.linspace(5, 50, 10)
        msd = np.linspace(5, 50, 10)
        dmsd = msd * 0.1
        diff = diffusion.Diffusion(dt, msd, dmsd)
        assert_equal(diff.function, straight_line)
        assert_almost_equal(diff.abscissa.m, dt)
        assert_equal(diff.abscissa.u, UREG.femtosecond)
        assert_almost_equal(diff.y_n, msd)
        assert_almost_equal(diff.y_s, dmsd)
        assert_equal(diff.ordinate.u, UREG.angstrom**2)
 
    def test_diffusion_D_max_likelihood(self):
        """
        Test the max likelihood of diffusion
        """
        dt = np.linspace(5, 50, 10)
        msd = np.linspace(5, 50, 10)
        dmsd = msd * 0.1
        diff = diffusion.Diffusion(dt, msd, dmsd)
        diff.max_likelihood()
        assert_almost_equal(diff.diffusion_coefficient.m, 1 / 6)

    def test_diffusion_D_mcmc_a(self):
        """
        Test the mcmc of diffusion
        """
        dt = np.linspace(5, 50, 10)
        msd = np.linspace(5, 50, 10)
        dmsd = msd * 0.1
        diff = diffusion.Diffusion(dt, msd, dmsd)
        diff.max_likelihood()
        diff.sample(n_samples=10, n_burn=10, progress=False)
        assert_equal(isinstance(diff.diffusion_coefficient, Distribution), True)
        assert_equal(diff.diffusion_coefficient.size, 1000)
        assert_equal(diff.variables[0].samples.min() > 0, True)
        assert_equal(len(diff.variables), 2)
    
    def test_diffusion_D_mcmc_b(self):
        """
        Test the mcmc of diffusion with unaccounted uncertainty
        """
        dt = np.linspace(5, 50, 10)
        msd = np.linspace(5, 50, 10)
        dmsd = msd * 0.1
        diff = diffusion.Diffusion(dt, msd, dmsd, unaccounted_uncertainty=True)
        diff.max_likelihood()
        diff.sample(n_samples=10, n_burn=10, progress=False)
        assert_equal(isinstance(diff.diffusion_coefficient, Distribution), True)
        assert_equal(diff.diffusion_coefficient.size, 1000)
        assert_equal(diff.variables[0].samples.min() > 0, True)
        assert_equal(len(diff.variables), 3)

    def test_diffusion_nested(self):
        """
        Test the nested method
        """
        dt = np.linspace(5, 50, 10)
        msd = np.linspace(5, 50, 10)
        dmsd = msd * 0.1
        diff = diffusion.Diffusion(dt, msd, dmsd, unaccounted_uncertainty=True)
        diff.max_likelihood()
        diff.nested(maxiter=10)
        assert_equal(diff.ln_evidence != None, True)
        assert_equal(isinstance(diff.ln_evidence, uncertainties.core.Variable), True)