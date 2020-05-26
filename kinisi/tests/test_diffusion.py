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
from scipy.stats import norm
from kinisi import diffusion
from uravu.distribution import Distribution
from uravu.utils import straight_line


dt = np.linspace(5, 50, 10)
msd = np.linspace(5, 50, 10)
MSD = []
for i in msd:
    MSD.append(Distribution(norm.rvs(loc=i, scale=i*0.1, size=5000, random_state=np.random.RandomState(1))))

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
        boot = diffusion.MSDBootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=100,
            max_resamples=1000)
        assert_equal(boot.dt.size, 5)
        assert_equal(len(boot.distributions), 5)
        assert_equal(boot.msd_observed.size, 5)
        assert_equal(boot.msd_sampled.size, 5)

    def test_msd_bootstrap_b(self):
        """
        Test msd_bootstrap for initial uniform with progress.
        """
        ordinate = np.random.uniform(size=(100, 50, 3))
        to_resample = [
            np.array(ordinate) for i in range(1, 6)
        ]
        boot = diffusion.MSDBootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=True,
            n_resamples=1,
            max_resamples=1000)
        assert_equal(boot.dt.size, 5)
        assert_equal(len(boot.distributions), 5)
        assert_equal(boot.msd_observed.size, 5)
        assert_equal(boot.msd_sampled.size, 5)

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
        boot = diffusion.MSDBootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=1,
            max_resamples=10, 
            sub_sample_dt=2)
        assert_equal(boot.dt.size, 3)
        assert_equal(len(boot.distributions), 3)
        assert_equal(boot.msd_observed.size, 3)
        assert_equal(boot.msd_sampled.size, 3)
    
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
        boot = diffusion.MSDBootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=1,
            max_resamples=10, 
            sub_sample_dt=2)
        assert_equal(boot.dt.size, 1)
        assert_equal(len(boot.distributions), 1)
        assert_equal(boot.msd_observed.size, 1)
        assert_equal(boot.msd_sampled.size, 1)

    def test_mscd_bootstrap_a(self):
        """
        Test mscd_bootstrap for initial normal.
        """
        ordinate = np.random.randn(100, 50, 3)
        to_resample = [
            np.array(ordinate) for i in range(1, 6)
        ]
        boot = diffusion.MSCDBootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=100,
            max_resamples=1000)
        assert_equal(boot.dt.size, 5)
        assert_equal(len(boot.distributions), 5)
        assert_equal(boot.msd_observed.size, 5)
        assert_equal(boot.msd_sampled.size, 5)

    def test_mscd_bootstrap_b(self):
        """
        Test mscd_bootstrap for initial uniform with progress.
        """
        ordinate = np.random.uniform(size=(100, 50, 3))
        to_resample = [
            np.array(ordinate) for i in range(1, 6)
        ]
        boot = diffusion.MSCDBootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=True,
            n_resamples=1,
            max_resamples=1000)
        assert_equal(boot.dt.size, 5)
        assert_equal(len(boot.distributions), 5)
        assert_equal(boot.msd_observed.size, 5)
        assert_equal(boot.msd_sampled.size, 5)

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
        boot = diffusion.MSCDBootstrap(
            np.linspace(100, 600, 5, dtype=int),
            to_resample,
            progress=False,
            n_resamples=1,
            max_resamples=10, 
            sub_sample_dt=2)
        assert_equal(boot.dt.size, 2)
        assert_equal(len(boot.distributions), 2)
        assert_equal(boot.msd_observed.size, 2)
        assert_equal(boot.msd_sampled.size, 2)

    def test_diffusion_init(self):
        """
        Test the initialisation of diffusion
        """
        bnds = ((0, 1000), (0, 1000))
        diff = diffusion.Diffusion(dt, MSD, bnds)
        assert_equal(diff.function, straight_line)
        assert_almost_equal(diff.abscissa, dt)
        assert_almost_equal(diff.y.n, msd, decimal=0)
        assert_almost_equal(diff.y.s, np.array([msd*0.196, msd*0.196]), decimal=0)
 
    def test_diffusion_D_max_likelihood(self):
        """
        Test the max likelihood of diffusion
        """
        bnds = ((0, 1000), (0, 1000))
        diff = diffusion.Diffusion(dt, MSD, bnds)
        diff.max_likelihood('mini')
        assert_almost_equal(diff.diffusion_coefficient.n, 1 / 6, decimal=1)

    def test_diffusion_D_mcmc(self):
        """
        Test the mcmc of diffusion
        """
        bnds = ((0, 1000), (0, 1000))
        diff = diffusion.Diffusion(dt, MSD, bnds)
        diff.max_likelihood('mini')
        diff.mcmc(n_samples=10, n_burn=10, progress=False)
        assert_equal(isinstance(diff.diffusion_coefficient, Distribution), True)
        assert_equal(diff.diffusion_coefficient.size, 500)
        assert_equal(diff.variables[0].samples.min() > 0, True)
        assert_equal(len(diff.variables), 2)
    
    def test_diffusion_nested(self):
        """
        Test the nested method
        """
        bnds = ((0, 2), (0, 2))
        diff = diffusion.Diffusion(dt, MSD, bnds)
        diff.max_likelihood('mini')
        diff.nested_sampling(maxiter=100)
        assert_equal(diff.ln_evidence != None, True)
        assert_equal(isinstance(diff.ln_evidence, uncertainties.core.Variable), True)