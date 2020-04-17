"""
Tests for arrhenius module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
import uncertainties
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

    def test_all_positive_priors(self):
        """
        Test the creation of an all positive prior.
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.StandardArrhenius(temp, ea, dea)
        arr.max_likelihood()
        priors = arr.all_positive_prior()
        assert_equal(len(priors), 2)
        assert_equal(priors[0].pdf(-1), 0)
        assert_equal(priors[0].pdf(1000), 0)
        assert_equal(priors[0].pdf(arr.variable_medians[0]), 1 / (arr.variable_medians[0] * 2 + arr.variable_medians[0]))
        assert_equal(priors[1].pdf(-1), 0)
        assert_equal(priors[1].pdf(1000), 0)
        assert_equal(priors[1].pdf(arr.variable_medians[1]), 1 / (arr.variable_medians[1] * 2 + arr.variable_medians[1]))

    def test_all_positive_priors_with_uu(self):
        """
        Test the creation of an all positive prior.
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.StandardArrhenius(temp, ea, dea, unaccounted_uncertainty=True)
        arr.max_likelihood()
        priors = arr.all_positive_prior()
        assert_equal(len(priors), 3)
        assert_equal(priors[0].pdf(-1), 0)
        assert_equal(priors[0].pdf(1000), 0)
        assert_equal(priors[0].pdf(arr.variable_medians[0]), 1 / (arr.variable_medians[0] * 2 + arr.variable_medians[0]))
        assert_equal(priors[1].pdf(-1), 0)
        assert_equal(priors[1].pdf(1000), 0)
        assert_equal(priors[1].pdf(arr.variable_medians[1]), 1 / (arr.variable_medians[1] * 2 + arr.variable_medians[1]))
        assert_equal(priors[2].pdf(-100), 0)
        assert_equal(priors[2].pdf(1000), 0)
        assert_equal(priors[2].pdf(arr.variable_medians[2]), 1 / (11))

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

    def test_all_positive_priors_super(self):
        """
        Test the creation of an all positive prior.
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.SuperArrhenius(temp, ea, dea)
        arr.max_likelihood()
        priors = arr.all_positive_prior()
        assert_equal(len(priors), 3)
        assert_equal(priors[0].pdf(-1), 0)
        assert_equal(priors[0].pdf(100000), 0)
        assert_equal(priors[0].pdf(arr.variable_medians[0]), 1 / (arr.variable_medians[0] * 4 + arr.variable_medians[0]))
        assert_equal(priors[1].pdf(-1), 0)
        assert_equal(priors[1].pdf(100000), 0)
        assert_equal(priors[1].pdf(arr.variable_medians[1]), 1 / (arr.variable_medians[1] * 4 + arr.variable_medians[1]))
        assert_equal(priors[2].pdf(-1), 0)
        assert_equal(priors[2].pdf(5), 0)
        assert_equal(priors[2].pdf(2.5), 1 / (4.9))

    def test_all_positive_priors_super_with_uu(self):
        """
        Test the creation of an all positive prior.
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.SuperArrhenius(temp, ea, dea, unaccounted_uncertainty=True)
        arr.max_likelihood()
        priors = arr.all_positive_prior()
        assert_equal(len(priors), 4)
        assert_equal(priors[0].pdf(-1), 0)
        assert_equal(priors[0].pdf(100000), 0)
        assert_equal(priors[0].pdf(arr.variable_medians[0]), 1 / (arr.variable_medians[0] * 4 + arr.variable_medians[0]))
        assert_equal(priors[1].pdf(-1), 0)
        assert_equal(priors[1].pdf(100000), 0)
        assert_equal(priors[1].pdf(arr.variable_medians[1]), 1 / (arr.variable_medians[1] * 4 + arr.variable_medians[1]))
        assert_equal(priors[2].pdf(-1), 0)
        assert_equal(priors[2].pdf(5), 0)
        assert_equal(priors[2].pdf(2.5), 1 / (4.9))
        assert_equal(priors[3].pdf(-100), 0)
        assert_equal(priors[3].pdf(1000), 0)
        assert_equal(priors[3].pdf(-1), 1 / (11))

    def test_standard_arrhenius(self):
        """
        Test the arrhenius function
        """
        assert_almost_equal(1.353352832, arrhenius.arrhenius(300, 4988.67588, 10), decimal=5)

    def test_super_arrhenius(self):
        """
        Test the super arrhenius function
        """
        assert_almost_equal(1.353352832, arrhenius.super_arrhenius(300, 4822.386684, 10, 10), decimal=5)

    def test_arrhenius_mcmc(self):
        """
        Test the mcmc of arrhenius
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.StandardArrhenius(temp, ea, dea)
        arr.max_likelihood()
        arr.sample(n_samples=10, n_burn=10, progress=False)
        assert_equal(isinstance(arr.variables[0], Distribution), True)
        assert_equal(arr.variables[0].size, 1000)
        assert_equal(arr.variables[0].samples.min() > 0, True)
        assert_equal(len(arr.variables), 2)

    def test_diffusion_D_mcmc_a(self):
        """
        Test the mcmc of super arrhenius
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.SuperArrhenius(temp, ea, dea)
        arr.max_likelihood()
        arr.sample(n_samples=10, n_burn=10, progress=False)
        assert_equal(isinstance(arr.variables[0], Distribution), True)
        assert_equal(arr.variables[0].size, 1000)
        assert_equal(arr.variables[0].samples.min() > 0, True)
        assert_equal(len(arr.variables), 3)

    def test_arrhenius_nested(self):
        """
        Test the nested method
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.StandardArrhenius(temp, ea, dea)
        arr.max_likelihood()
        arr.nested(maxiter=10)
        assert_equal(arr.ln_evidence != None, True)
        assert_equal(isinstance(arr.ln_evidence, uncertainties.core.Variable), True)

    def test_arrhenius_nested_super(self):
        """
        Test the nested method
        """
        temp = np.linspace(5, 50, 10)
        ea = np.linspace(5, 50, 10)
        dea = ea * 0.1
        arr = arrhenius.SuperArrhenius(temp, ea, dea)
        arr.max_likelihood()
        arr.nested(maxiter=10)
        assert_equal(arr.ln_evidence != None, True)
        assert_equal(isinstance(arr.ln_evidence, uncertainties.core.Variable), True)