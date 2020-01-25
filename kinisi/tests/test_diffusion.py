"""
Tests for msd module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from kinisi.diffusion import Diffusion
from kinisi.distribution import Distribution


class TestMsd(unittest.TestCase):
    """
    Unit tests for utils module
    """
    def test_msd_init_a(self):
        """
        Test the initialisation of the MSD class with defaults.
        """
        sq_disp = [np.ones((i, i+1)) for i in range(1, 6)[::-1]]
        diff = Diffusion(sq_disp, np.linspace(1, 100, len(sq_disp)))
        for i, disp in enumerate(sq_disp):
            assert_almost_equal(diff.displacements[i], disp)
        assert_equal(diff.ordinate, np.ones((5)))
        num_part = np.array([(i * (i+1)) for i in range(1, 6)[::-1]])
        assert_equal(diff.ordinate_error, np.sqrt(6 / num_part))

    def test_msd_init_b(self):
        """
        Test the initialisation of the MSD class without defaults.
        """
        sq_disp = [np.ones((i, i+1)) for i in range(1, 6)[::-1]]
        diff = Diffusion(
            sq_disp,
            np.linspace(1, 100, len(sq_disp)),
            step_freq=2,
        )
        expected_sq_disp = sq_disp[::2]
        for i, disp in enumerate(expected_sq_disp):
            assert_almost_equal(diff.displacements[i], disp)
        assert_equal(diff.ordinate, np.ones((3)))
        num_part = np.array([(i * (i+1)) for i in range(1, 6)[::-2]])
        assert_equal(diff.ordinate_error, np.sqrt(6 / num_part))

    def test_resample(self):
        """
        Test resample with default confidence intervals.
        """
        data = [np.ones((5, 5))] * 5
        diff = Diffusion(data, np.linspace(1, 10, 5))
        diff.resample(progress=False)
        assert_equal(diff.ordinate.size, 5)
        assert_equal(diff.ordinate_error.size, 5)
        assert_equal(diff.resampled, True)

    def test_sample_diffusion(self):
        """
        Test sample_diffusion.
        """
        data = [np.ones((5, 5))] * 5
        diff = Diffusion(data, np.linspace(1, 10, 5))
        diff.sample_diffusion(
            n_samples=5,
            n_burn=5,
            progress=False,
        )
        assert_equal(diff.mcmced, True)
        assert_equal(isinstance(diff.gradient, Distribution), True)
        assert_equal(diff.gradient.size, 500)
        assert_equal(isinstance(diff.intercept, Distribution), True)
        assert_equal(diff.intercept.size, 500)
        assert_equal(
            isinstance(diff.diffusion_coefficient, Distribution),
            True,
        )
        assert_equal(diff.diffusion_coefficient.size, 500)
