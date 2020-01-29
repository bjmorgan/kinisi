"""
Tests for utils module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from kinisi import relationships


class TestUtils(unittest.TestCase):
    """
    Unit tests for utils module
    """
    def test_lnl(self):
        """
        Test lnl function.
        """
        y_data = np.ones(10)
        dy_data = np.ones(10) * 0.1
        model = np.ones(10) * 1.1
        expected_lnl = 8.8364655979
        actual_lnl = relationships.lnl(model, y_data, dy_data)
        assert_almost_equal(actual_lnl, expected_lnl)
