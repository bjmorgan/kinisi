"""
Tests for analyze module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

import os
import warnings
import unittest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from kinisi.matrix import find_nearest_positive_definite, check_positive_definite

class TestFunctions(unittest.TestCase):
    """
    Tests for the matrix functions.
    """
    def test_find_psd_matrix(self):
        start_matrix = np.eye(20)
        assert_equal(start_matrix, find_nearest_positive_definite(start_matrix))

    def test_find_non_psd_matrix(self):
        start_matrix = np.array([[0.99, 0.78, 0.59, 0.44],
                                 [0.78, 0.92, 0.28, 0.81],
                                 [0.59, 0.28, 1.12, 0.23],
                                 [0.44, 0.81, 0.23, 0.99]])
        result = find_nearest_positive_definite(start_matrix)
        assert_raises(AssertionError, assert_almost_equal, start_matrix, result)

    def test_check_psd_matrix(self):
        start_matrix = np.eye(20)
        assert check_positive_definite(start_matrix) == True

    def test_check_non_psd_matrix(self):
        start_matrix = np.array([[0.99, 0.78, 0.59, 0.44],
                                 [0.78, 0.92, 0.28, 0.81],
                                 [0.59, 0.28, 1.12, 0.23],
                                 [0.44, 0.81, 0.23, 0.99]])
        assert check_positive_definite(start_matrix) == False
 