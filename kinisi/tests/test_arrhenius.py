"""
Tests for arrhenius module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_equal, assert_almost_equal
from kinisi import arrhenius
from uravu.distribution import Distribution
from uravu.utils import straight_line

temp = np.linspace(5, 50, 10)
ea = np.linspace(5, 50, 10)
EA = []
for e in ea:
    EA.append(Distribution(norm.rvs(loc=e, scale=e*0.1, size=5000, random_state=np.random.RandomState(1))))


class TestArrhenius(unittest.TestCase):
    """
    Unit tests for arrhenius module
    """
    def test_standard_arrhenius_init(self):
        """
        Test the initialisation of standard arrhenius
        """
        arr = arrhenius.StandardArrhenius(temp, EA, ((0, 1000), (0, 1000)))
        assert_equal(arr.function, arrhenius.arrhenius)
        assert_almost_equal(arr.abscissa, temp)
        assert_almost_equal(arr.y.n, ea, decimal=0)
        assert_almost_equal(arr.y.s, np.array([ea*0.196, ea*0.196]), decimal=0)

    def test_super_arrhenius_init(self):
        """
        Test the initialisation of super arrhenius
        """
        arr = arrhenius.SuperArrhenius(temp, EA, ((0, 1000), (0, 1000), (0, 4)))
        assert_equal(arr.function, arrhenius.super_arrhenius)
        assert_almost_equal(arr.abscissa, temp)
        assert_almost_equal(arr.y.n, ea, decimal=0)
        assert_almost_equal(arr.y.s, np.array([ea*0.196, ea*0.196]), decimal=0)

    def test_standard_arrhenius(self):
        """
        Test the arrhenius function
        """
        assert_almost_equal(9.996132574, arrhenius.arrhenius(300, 1e-5, 10), decimal=5)

    def test_super_arrhenius(self):
        """
        Test the super arrhenius function
        """
        assert_almost_equal(9.995999241, arrhenius.super_arrhenius(300, 1e-5, 10, 10), decimal=5)
