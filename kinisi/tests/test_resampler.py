"""
Tests for resampler module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from kinisi import resampler


class TestDistribution(unittest.TestCase):
    """
    Unit tests for resample module.
    """
    def test_init_a(self):
        """
        Test initialisation with all defaults
        """
        displacements = [np.ones((1, 1))]
        dist = resampler.Distribution(displacements)

        assert_almost_equal(dist.displacements, [np.ones((1, 1))])
        assert_equal(dist.resamples, 2000)
        assert_equal(dist.samples_freq, 1)
        assert_almost_equal(dist.median, np.zeros((1)))
        assert_almost_equal(dist.confidence_interval, [2.5, 97.5])
        assert_almost_equal(dist.span, np.zeros((1, 2)))

    def test_init_b(self):
        """
        Test initialisation with non-defaults
        """
        displacements = [np.ones((1, 1)) for i in range(10)]
        dist = resampler.Distribution(
            displacements,
            resamples=10,
            samples_freq=3,
            step_freq=2,
            confidence_interval=[5, 95]
        )

        assert_almost_equal(
            dist.displacements,
            [np.ones((1, 1)) for i in range(5)]
        )
        assert_equal(dist.resamples, 10)
        assert_equal(dist.samples_freq, 3)
        assert_almost_equal(dist.median, np.zeros((5)))
        assert_almost_equal(dist.confidence_interval, [5., 95.])
        assert_almost_equal(dist.span, np.zeros((5, 2)))
