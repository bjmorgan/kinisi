"""
Tests for the displacement module.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)
# pylint: disable=R0201

import unittest

import numpy as np
import pytest
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
        disp = sc.Variable(values=np.arange(0, 12, 1).reshape((4, 3, 1)), dims=['obs', 'particle', 'dimension'])
        expected = sc.Variable(
            values=np.array([3, 12, 21, 30]).reshape((4, 1, 1)), dims=['obs', 'particle', 'dimension']
        )
        actual = displacement._consolidate_system_particles(disp)
        testing.assert_identical(actual, expected)

    def test_consolidate_system_particles_multiple_system_particles(self):
        """
        Test the consolidation of system particles where the number of system particles is greater than 1
        and divides nicely by the number of atoms.
        """
        disp = sc.Variable(values=np.arange(0, 24, 1).reshape((4, 6, 1)), dims=['obs', 'particle', 'dimension'])
        expected = sc.Variable(
            values=np.array([3, 12, 21, 30, 39, 48, 57, 66]).reshape((4, 2, 1)), dims=['obs', 'particle', 'dimension']
        )
        actual = displacement._consolidate_system_particles(disp, system_particles=2)
        testing.assert_identical(actual, expected)

    def test_consolidate_system_particles_non_divisible_system_particles(self):
        """
        Test the consolidation of system particles where the number of system particles does not divide nicely
        by the number of atoms.
        """
        with pytest.warns(UserWarning) as record:
            disp = sc.Variable(values=np.arange(0, 28, 1).reshape((4, 7, 1)), dims=['obs', 'particle', 'dimension'])
            expected = sc.Variable(
                values=np.array([3, 12, 24, 33, 45, 54, 66, 75]).reshape((4, 2, 1)),
                dims=['obs', 'particle', 'dimension'],
            )
            actual = displacement._consolidate_system_particles(disp, system_particles=2)
            print(actual.values)
            print(expected.values)
            testing.assert_identical(actual, expected)
        assert len(record) == 1
        assert str(record[0].message) == (
            'Truncating 7 atoms to split evenly into 2 centres of mass. This approach '
            + 'is inefficient, you should consider using the number of system particles '
            + 'to split this evenly.'
        )


class LimitedParser:
    def __init__(self, displacements, dt, time_step, step_skip):
        self.displacements = displacements
        self.dt = dt
        self.dt_index = (dt / (time_step * step_skip)).astype(int)
        self.dimensionality = displacements.sizes['dimension'] * sc.units.dimensionless


displacements = sc.Variable(
    values=np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]).reshape((4, 2, 1)), dims=['obs', 'particle', 'dimension']
)
dt = sc.Variable(values=np.arange(1, 5, 1), dims=['time interval'], unit='fs')
time_step = 1 * sc.Unit('fs')
step_skip = 1 * sc.Unit('dimensionless')
PARSER = LimitedParser(displacements, dt, time_step, step_skip)


class TestCalculateMsd(unittest.TestCase):
    """
    Unit tests for the calculate_msd function in the displacement module.
    """

    def test_calculate_msd(self):
        """
        Test the calculation of the mean-squared displacement.
        """
        result = displacement.calculate_msd(PARSER)
        expected = sc.DataArray(
            data=sc.Variable(
                dims=['time interval'],
                values=[1.15625, 4.375, 9.8125, 18.125],
                variances=[0.0244140625, 0.2109375, 0.990234375, 4.515625],
            ),
            coords={
                'time interval': PARSER.dt,
                'n_samples': sc.array(values=[8.0, 4.0, 8 / 3, 2], dims=['time interval'], unit='dimensionless'),
            },
        )
        testing.assert_allclose(result['da'], expected)

    def test_calculate_msd_progress(self):
        """
        Test the calculation of the mean-squared displacement with progress.
        """
        result = displacement.calculate_msd(PARSER, progress=True)
        expected = sc.DataArray(
            data=sc.Variable(
                dims=['time interval'],
                values=[1.15625, 4.375, 9.8125, 18.125],
                variances=[0.0244140625, 0.2109375, 0.990234375, 4.515625],
            ),
            coords={
                'time interval': PARSER.dt,
                'n_samples': sc.array(values=[8.0, 4.0, 8 / 3, 2], dims=['time interval'], unit='dimensionless'),
            },
        )
        testing.assert_allclose(result['da'], expected)


class TestCalculateMstd(unittest.TestCase):
    """
    Unit tests for the calculate_mstd function in the displacement module.
    """

    def test_calculate_mstd(self):
        """
        Test the calculation of the mean-squared displacement.
        """
        result = displacement.calculate_mstd(PARSER)
        expected = sc.DataArray(
            data=sc.Variable(
                dims=['time interval'],
                values=[2.28125, 8.7083335, 19.5625],
                variances=[0.0791015625, 0.7526041675, 3.662109375],
            ),
            coords={
                'time interval': PARSER.dt['time interval', :-1],
                'n_samples': sc.array(values=[8 / 2, 4 / 2, 8 / 6], dims=['time interval'], unit='dimensionless'),
            },
        )
        testing.assert_allclose(result['da'], expected)

    def test_calculate_mstd_progress(self):
        """
        Test the calculation of the mean-squared displacement with progress.
        """
        result = displacement.calculate_mstd(PARSER, progress=True)
        expected = sc.DataArray(
            data=sc.Variable(
                dims=['time interval'],
                values=[2.28125, 8.7083335, 19.5625],
                variances=[0.0791015625, 0.7526041675, 3.662109375],
            ),
            coords={
                'time interval': PARSER.dt['time interval', :-1],
                'n_samples': sc.array(values=[8 / 2, 4 / 2, 8 / 6], dims=['time interval'], unit='dimensionless'),
            },
        )
        testing.assert_allclose(result['da'], expected)

    def test_calculate_mstd_ionic_charge(self):
        """
        Test the calculation of the mean-squared displacement with some ionic charge.
        """
        result = displacement.calculate_mstd(PARSER, ionic_charge=2 * sc.Unit('elementary_charge'), progress=True)
        per_ion = displacement.calculate_mstd(
            PARSER,
            ionic_charge=sc.array(dims=['particle'], values=[2.0, 2.0], unit='elementary_charge'),
            progress=False,
        )
        testing.assert_allclose(per_ion['da'], result['da'])
        expected = sc.DataArray(
            data=sc.Variable(
                dims=['time interval'],
                values=[9.125, 34.833333335, 78.25],
                variances=[1.265625, 12.0416666675, 58.59375],
                unit=sc.Unit('elementary_charge') ** 2,
            ),
            coords={
                'time interval': PARSER.dt['time interval', :-1],
                'n_samples': sc.array(values=[8 / 2, 4 / 2, 8 / 6], dims=['time interval'], unit='dimensionless'),
            },
        )
        testing.assert_allclose(result['da'], expected)


def test_calculate_mstd_mixed_ionic_charge():
    """
    Charges of opposite sign must be applied to each ion before the displacements are summed.
    """
    charges = sc.array(dims=['particle'], values=[1.0, -1.0], unit='elementary_charge')
    result = displacement.calculate_mstd(PARSER, ionic_charge=charges, progress=False)
    signs = sc.array(dims=['particle'], values=[1.0, -1.0])
    flipped = LimitedParser(displacements * signs, dt, time_step, step_skip)
    expected = displacement.calculate_mstd(flipped, ionic_charge=1 * sc.Unit('elementary_charge'), progress=False)
    testing.assert_allclose(result['da'], expected['da'])
