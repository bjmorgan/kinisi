"""
Tests for distribution module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from kinisi.distribution import Distribution


class TestDistribution(unittest.TestCase):
    """
    Testing the Distribution class.
    """
    def test_init_a(self):
        """
        Test initialisation with defaults.
        """
        distro = Distribution()
        assert_equal(distro.samples, np.array([]))
        assert_equal(distro.median, None)
        assert_equal(distro.con_int, None)
        assert_almost_equal(distro.ci_points, [2.5, 97.5])

    def test_init_b(self):
        """
        Test initialisation without defaults.
        """
        distro = Distribution([5., 95.])
        assert_equal(distro.samples, np.array([]))
        assert_equal(distro.median, None)
        assert_equal(distro.con_int, None)
        assert_almost_equal(distro.ci_points, [5., 95.])

    def test_check_normality_less_than_5000(self):
        """
        Test check_normality with less than 5000 samples.
        """
        distro = Distribution()
        np.random.seed(1)
        distro.add_samples(np.random.randn(1000))
        assert_equal(distro.normal, True)

    def test_check_normality_more_than_5000(self):
        """
        Test check_normality with more than 5000 samples.
        """
        distro = Distribution()
        np.random.seed(1)
        distro.add_samples(np.random.randn(10000))
        assert_equal(distro.normal, True)
