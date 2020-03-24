"""
Tests for parser module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

# This parser borrows heavily from the
# pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer
# class, originally authored by Will Richards
# (wrichard@mit.edu) and Shyue Ping Ong.
# We include this statement to not that we make
# no claim to authorship of that code and make
# no attack on the original authors.
#
# In fact, we love pymatgen!

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from pymatgen.core import Structure
from kinisi import parser
from uravu.distribution import Distribution
from uravu.utils import straight_line


class TestParser(unittest.TestCase):
    """
    Unit tests for parser module
    """
    def test_standard_arrhenius_init(self):
        """
        Test the initialisation of diffusion
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
