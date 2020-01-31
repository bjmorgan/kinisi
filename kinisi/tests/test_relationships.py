"""
Tests for utils module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
import uncertainties
from numpy.testing import assert_almost_equal, assert_equal
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

    def test_nested_sampling(self):
        """
        Test nested sampling.
        """
        y_data = np.ones(10) * np.random.random(10)
        dy_data = np.ones(10) * 0.1
        x_data = np.linspace(1, 10, 10)
        rel = relationships.Relationship(relationships.straight_line, x_data, y_data, dy_data)
        rel.variables = rel.fit([1, 0], with_uncertainty=False)
        rel.nested_sampling(progress=False, maxiter=10)
        assert_equal(isinstance(rel.evidence, uncertainties.core.Variable), True)

    def test_straight_line_b(self):
        """
        Test StraightLine class with variables names
        """
        y_data = np.linspace(1, 10, 10) * 0.01 * np.random.random(10)
        dy_data = np.ones(10) * 0.01
        x_data = np.linspace(1, 10, 10)
        arr = relationships.StraightLine(x_data, y_data, dy_data, variable_name=[r'a', r'b'])
        assert_equal(arr.abscissa, x_data)
        assert_equal(arr.ordinate, y_data)
        assert_equal(arr.ordinate_error, dy_data)
        assert_equal(arr.variables_name, [r'a', r'b'])

    def test_arrhenius_a(self):
        """
        Test Arrhenius class.
        """
        y_data = np.ones(10) * np.random.random(10)
        dy_data = np.ones(10) * 0.1
        x_data = np.linspace(1, 10, 10)
        arr = relationships.Arrhenius(x_data, y_data, dy_data)
        assert_equal(arr.abscissa, x_data)
        assert_equal(arr.ordinate, y_data)
        assert_equal(arr.ordinate_error, dy_data)
        assert_equal(arr.variables_name, [r'$E_a$', r'$A$'])

    def test_arrhenius_b(self):
        """
        Test Arrhenius class with variables names
        """
        y_data = np.ones(10) * np.random.random(10)
        dy_data = np.ones(10) * 0.1
        x_data = np.linspace(1, 10, 10)
        arr = relationships.Arrhenius(x_data, y_data, dy_data, variable_name=[r'a', r'b'])
        assert_equal(arr.abscissa, x_data)
        assert_equal(arr.ordinate, y_data)
        assert_equal(arr.ordinate_error, dy_data)
        assert_equal(arr.variables_name, [r'a', r'b'])

    def test_vtfequation_a(self):
        """
        Test VTFEquation class.
        """
        x_data = np.linspace(1, 10, 10)
        y_data = np.exp(x_data) * np.random.random(10)
        dy_data = np.ones(10) * 0.1
        vtf = relationships.VTFEquation(x_data, y_data, dy_data)
        assert_equal(vtf.abscissa, x_data)
        assert_equal(vtf.ordinate, y_data)
        assert_equal(vtf.ordinate_error, dy_data)
        assert_equal(vtf.variables_name, [r'$E_a$', r'$A$', r'$T_0$'])

    def test_vtfequation_b(self):
        """
        Test VTFEquation class with variables names
        """
        x_data = np.linspace(1, 10, 10)
        y_data = np.exp(x_data) * np.random.random(10)
        dy_data = np.ones(10) * 0.1
        vtf = relationships.VTFEquation(x_data, y_data, dy_data, variable_name=[r'a', r'b', r'c'])
        assert_equal(vtf.abscissa, x_data)
        assert_equal(vtf.ordinate, y_data)
        assert_equal(vtf.ordinate_error, dy_data)
        assert_equal(vtf.variables_name, [r'a', r'b', r'c'])
