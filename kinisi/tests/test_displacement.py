"""
Tests for the displacement module.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)
# pylint: disable=R0201

import unittest
import numpy as np
import scipp as sc
from scipp import testing
from kinisi import displacement


class TestSystemParticleConsoldiation(unittest.TestCase):
    """
    Unit tests for the _consolidate_system_particles function in the displacement module.
    """

    def test_consolidate_system_particles_one_system_particle(self):
        """
        Test the consolidation of system particles where the number of system particles is 1.
        """
        disp = sc.Variable(values=np.arange(0, 12, 1).reshape((4, 3, 1)), dims=['obs', 'atom', 'dimension'])
        expected = sc.Variable(values=np.array([3, 12, 21, 30]).reshape((4, 1, 1)), dims=['obs', 'atom', 'dimension'])
        actual = displacement._consolidate_system_particles(disp)
        testing.assert_identical(actual, expected)

    def test_consolidate_system_particles_multiple_system_particles(self):
        """
        Test the consolidation of system particles where the number of system particles is greater than 1
        and divides nicely by the number of atoms.
        """
        disp = sc.Variable(values=np.arange(0, 24, 1).reshape((4, 6, 1)), dims=['obs', 'atom', 'dimension'])
        expected = sc.Variable(values=np.array([3, 12, 21, 30, 39, 48, 57, 66]).reshape((4, 2, 1)),
                               dims=['obs', 'atom', 'dimension'])
        actual = displacement._consolidate_system_particles(disp, system_particles=2)
        testing.assert_identical(actual, expected)

    def test_consolidate_system_particles_non_divisible_system_particles(self):
        """
        Test the consolidation of system particles where the number of system particles does not divide nicely
        by the number of atoms.
        """
        disp = sc.Variable(values=np.arange(0, 28, 1).reshape((4, 7, 1)), dims=['obs', 'atom', 'dimension'])
        expected = sc.Variable(values=np.array([3, 12, 24, 33, 45, 54, 66, 75]).reshape((4, 2, 1)),
                               dims=['obs', 'atom', 'dimension'])
        actual = displacement._consolidate_system_particles(disp, system_particles=2)
        testing.assert_identical(actual, expected)
