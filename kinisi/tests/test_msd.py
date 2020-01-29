"""
Tests for msd module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_equal
from kinisi import msd


class TestMsd(unittest.TestCase):
    """
    Unit tests for msd module
    """

    def test_bootstrap_a(self):
        """
        Test bootstrap for initial normal.
        """
        ordinate = np.random.randn((10))
        to_resample = [
            np.array([ordinate[:-i], ordinate[:-i]]) for i in range(1, 6)
        ]
        mean, err = msd.bootstrap(to_resample, progress=False, n_resamples=10)
        assert_equal(mean.size, 5)
        assert_equal(err.size, 5)
