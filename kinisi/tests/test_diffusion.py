"""
Tests for msd module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from kinisi import diffusion, UREG
from kinisi.distribution import Distribution


class TestMsd(unittest.TestCase):
    """
    Unit tests for msd module
    """

    def test_msd_bootstrap(self):
        """
        Test msd_bootstrap for initial normal.
        """
        ordinate = np.random.randn((10))
        to_resample = [
            [
                [ordinate, ordinate],
                [ordinate, ordinate]
            ],
            [
                [ordinate, ordinate],
                [ordinate, ordinate]
            ]
        ]
        to_resample = [
            np.array(to_resample) for i in range(1, 6)
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.msd_bootstrap(
            np.arange(1, 6),
            to_resample,
            progress=False,
            n_resamples=10,
            max_resamples=10)
        assert_equal(delta_t.size, 5)
        assert_equal(mean.size, 5)
        assert_equal(err.size, 5)
        assert_equal(con_int_l.size, 5)
        assert_equal(con_int_u.size, 5)

    def test_mscd_bootstrap(self):
        """
        Test mscd_bootstrap for initial normal.
        """
        ordinate = np.random.randn((10))
        to_resample = [
            [
                [ordinate, ordinate],
                [ordinate, ordinate]
            ],
            [
                [ordinate, ordinate],
                [ordinate, ordinate]
            ]
        ]
        to_resample = [
            np.array(to_resample) for i in range(1, 6)
        ]
        delta_t, mean, err, con_int_l, con_int_u = diffusion.mscd_bootstrap(
            np.arange(1, 6),
            to_resample,
            [1],
            progress=False,
            n_resamples=10,
            max_resamples=10)
        assert_equal(delta_t.size, 5)
        assert_equal(mean.size, 5)
        assert_equal(err.size, 5)
        assert_equal(con_int_l.size, 5)
        assert_equal(con_int_u.size, 5)

    def test_diffusion_init(self):
        """
        Test for the initialisation of the Diffusion class.
        """
        mean = np.linspace(1, 10, 5)
        err = mean * 0.1
        abscissa = np.linspace(1, 10, 5)
        diff = diffusion.Diffusion(abscissa, mean, err)
        assert_almost_equal(diff.diffusion_coefficient.magnitude.n, 1 / 60)
        assert_almost_equal(diff.diffusion_coefficient.magnitude.s, 0)
        assert_equal(
            diff.diffusion_coefficient.units,
            UREG.centimeter ** 2 / UREG.second)
        assert_almost_equal(diff.abscissa, abscissa)
        assert_almost_equal(diff.ordinate, mean)
        assert_almost_equal(diff.ordinate_error, err)
        assert_equal(diff.abscissa_unit, UREG.femtosecond)
        assert_equal(diff.ordinate_unit, UREG.angstrom**2)
        assert_equal(diff.abscissa_name, r'$\delta t$')
        assert_equal(diff.ordinate_name, r'$\langle r ^ 2 \rangle$')

    def test_diffusion_mcmc(self):
        """
        Test the mcmc methd for the Diffusion class
        """
        mean = np.linspace(1, 10, 5) * np.random.randn(5)
        err = mean * 0.1
        abscissa = np.linspace(1, 10, 5)
        diff = diffusion.Diffusion(abscissa, mean, err)
        diff.sample(progress=False, n_burn=10, n_samples=10)
        assert_equal(diff.diffusion_coefficient.size, 1000)
        assert_equal(
            isinstance(
                diff.diffusion_coefficient, Distribution), True)
        assert_equal(
            diff.diffusion_coefficient.unit,
            UREG.centimeter ** 2 / UREG.second)
