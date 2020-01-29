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
        distro = Distribution([1])
        assert_equal(distro.samples, np.array([1]))
        assert_equal(distro.median, 1)
        assert_equal(distro.con_int, None)
        assert_almost_equal(distro.ci_points, [2.5, 97.5])

    def test_init_b(self):
        """
        Test initialisation without defaults.
        """
        distro = Distribution([1], ci_points=[5.0, 95.0])
        assert_equal(distro.samples, np.array([1]))
        assert_equal(distro.median, 1)
        assert_equal(distro.con_int, None)
        assert_almost_equal(distro.ci_points, [5.0, 95.0])

    def test_init_c(self):
        """
        Test initialisation with bad ci_points.
        """
        with self.assertRaises(ValueError):
            Distribution([1, 2], ci_points=[5.0, 95.0, 102.0])

    def test_check_normality_true_less_than_5000(self):
        """
        Test check_normality with less than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn(1000))
        assert_equal(distro.normal, True)
        assert_equal(distro.check_normality(), True)

    def test_check_normality_true_more_than_5000(self):
        """
        Test check_normality with more than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn(10000))
        assert_equal(distro.normal, True)
        assert_equal(distro.check_normality(), True)

    def test_check_normality_false_more_than_5000(self):
        """
        Test check_normality with more than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.rand(10000))
        assert_equal(distro.normal, False)
        assert_equal(distro.check_normality(), False)

    def test_check_normality_false_less_than_5000(self):
        """
        Test check_normality with less than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.rand(1000))
        assert_equal(distro.normal, False)
        assert_equal(distro.check_normality(), False)

    def test_check_normality_less_than_3(self):
        """
        Test check_normality with more than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn(2))
        assert_equal(distro.normal, False)
        assert_equal(distro.check_normality(), False)

    def test_add_samples_single(self):
        """
        Test add_samples with a single value.
        """
        distro = Distribution(1)
        assert_equal(distro.size, 1)
        assert_almost_equal(distro.samples, np.array([1]))
        assert_almost_equal(distro.median, 1)
        assert_equal(distro.error, None)
        assert_equal(distro.con_int, np.array([]))
        assert_equal(distro.normal, False)
