"""
Tests for utils module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from kinisi import utils
import unittest

class TestUtils(unittest.TestCase):
    """
    Unit tests for utils module
    """
    def test_straight_line_int(self):
        """
        Test straight line for int input
        """
        expected_y = np.linspace(3, 21, 10, dtype=int)        
        x = np.linspace(1, 10, 10, dtype=int)
        result_y = utils.straight_line(x, 2, 1)
        assert_equal(result_y.shape, expected_y.shape)
        assert_equal(result_y, expected_y)

    def test_straight_line_float(self):
        """
        Test straight line for float input
        """
        expected_y = np.linspace(3.5, 21.5, 10, dtype=float)        
        x = np.linspace(1.0, 10.0, 10, dtype=float)
        result_y = utils.straight_line(x, 2.0, 1.5)
        assert_equal(result_y.shape, expected_y.shape)
        assert_almost_equal(result_y, expected_y)
