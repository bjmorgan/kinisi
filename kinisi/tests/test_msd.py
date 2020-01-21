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
from kinisi.msd import MSD


class TestMsd(unittest.TestCase):
    """
    Unit tests for utils module
    """
    def test_msd_init_a(self):
        """
        Test the initialisation of the MSD class with defaults.
        """
        sq_disp = [np.ones((i, i+1)) for i in range(1, 6)[::-1]]
        msd = MSD(sq_disp)
        for i, disp in enumerate(sq_disp):
            assert_almost_equal(msd.sq_displacements[i], disp)
        assert_equal(msd.data, None)

    def test_msd_init_b(self):
        """
        Test the initialisation of the MSD class without defaults.
        """
        sq_disp = [np.ones((i, i+1)) for i in range(1, 6)[::-1]]
        msd = MSD(sq_disp, step_freq=2)
        expected_sq_disp = sq_disp[::2]
        for i, disp in enumerate(expected_sq_disp):
            assert_almost_equal(msd.sq_displacements[i], disp)
        assert_equal(msd.data, None)
