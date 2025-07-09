"""
Tests for parser module
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)
# pylint: disable=R0201

import unittest

import numpy as np

from kinisi import parser


class TestSubsetApprox(unittest.TestCase):
    """
    Unit tests for the subset approximation functionality.
    """

    def test_is_subset_approx(self):
        data = np.array([1, 2, 3, 4, 5])
        subset = np.array([1, 3, 5])
        assert parser.is_subset_approx(subset, data)

    def test_is_subset_approx_fail(self):
        data = np.array([1, 2, 3, 4, 5])
        subset = np.array([1, 3, 5, 7])
        assert not parser.is_subset_approx(subset, data)
