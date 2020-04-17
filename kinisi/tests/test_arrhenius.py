"""
Tests for arrhenius module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from kinisi import arrhenius, UREG
from uravu.distribution import Distribution
from uravu.utils import straight_line


class TestArrhenius(unittest.TestCase):
    """
    Unit tests for arrhenius module
    """
    def test_standard_arrhenius_init(self):
        """
        Test the initialisation of standard arrhenius
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.StandardArrhenius(temp, ea, dea)
        assert_equal(arr.function, arrhenius.arrhenius)
        assert_almost_equal(arr.abscissa.m, temp)
        assert_equal(arr.abscissa.u, UREG.kelvin)
        assert_almost_equal(arr.y_n, ea)
        assert_almost_equal(arr.y_s, dea)
        assert_equal(arr.ordinate.u, UREG.centimeter**2 / UREG.second)
        assert_equal(len(arr.variable_names), 2)
        assert_equal(len(arr.variable_units), 2)

    def test_standard_arrhenius_init_with_uu(self):
        """
        Test the initialisation of standard arrhenius with unaccounted uncertainty
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.StandardArrhenius(temp, ea, dea, unaccounted_uncertainty=True)
        assert_equal(arr.function, arrhenius.arrhenius)
        assert_almost_equal(arr.abscissa.m, temp)
        assert_equal(arr.abscissa.u, UREG.kelvin)
        assert_almost_equal(arr.y_n, ea)
        assert_almost_equal(arr.y_s, dea)
        assert_equal(arr.ordinate.u, UREG.centimeter**2 / UREG.second)
        assert_equal(len(arr.variable_names), 3)
        assert_equal(len(arr.variable_units), 3)

    def test_super_arrhenius_init(self):
        """
        Test the initialisation of super arrhenius
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.SuperArrhenius(temp, ea, dea)
        assert_equal(arr.function, arrhenius.super_arrhenius)
        assert_almost_equal(arr.abscissa.m, temp)
        assert_equal(arr.abscissa.u, UREG.kelvin)
        assert_almost_equal(arr.y_n, ea)
        assert_almost_equal(arr.y_s, dea)
        assert_equal(arr.ordinate.u, UREG.centimeter**2 / UREG.second) 

    def test_standard_arrhenius(self):
        """
        Test the arrhenius function
        """
        assert_almost_equal(1.353352832, arrhenius.arrhenius(300, 4988.67588, 10))

    def test_super_arrhenius(self):
        """
        Test the super arrhenius function
        """
        assert_almost_equal(1.353352832, arrhenius.super_arrhenius(300, 4822.386684, 10, 10))
