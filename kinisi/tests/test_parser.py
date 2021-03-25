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
import MDAnalysis as mda
from numpy.testing import assert_almost_equal, assert_equal
from pymatgen.core import Structure
from pymatgen.io.vasp import Xdatcar
import os
import kinisi
from kinisi import parser
from uravu.distribution import Distribution
from uravu.utils import straight_line


dc = np.random.random(size=(100, 100, 3))
indices = np.arange(0, 100, 1, dtype=int)
time_step = 1.0
step_skip = 1

class TestParser(unittest.TestCase):
    """
    Unit tests for parser module
    """
    def test_parser_init_timestep(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        assert_equal(p.time_step, time_step)

    def test_parser_init_stepskip(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        assert_equal(p.step_skip, step_skip)

    def test_parser_init_indices(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        assert_equal(p.indices, indices)

    def test_parser_init_min_dt(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        assert_equal(p.min_dt, 20)

    def test_parser_delta_t(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        assert_equal(p.delta_t.size, 80)

    def test_parser_disp_3d(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        assert_equal(len(p.disp_3d), 80)
        for i, d in enumerate(p.disp_3d):
            assert_equal(d.shape[0], 100)
            assert_equal(d.shape[1], 80-i)
            assert_equal(d.shape[2], 3)

    def test_smoothed_timesteps(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        timesteps = p.smoothed_timesteps(100, 30, indices)
        assert_equal(timesteps, np.arange(20, 100, 1))

    def test_smoothed_timesteps_not_enough(self):
        with self.assertRaises(ValueError):
            p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=120)
            p.smoothed_timesteps(100, 20, indices)

    def test_smoothed_timesteps_min_dt_zero(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=0)
        timesteps = p.smoothed_timesteps(100, 30, indices)
        assert_equal(timesteps, np.arange(1, 100, 1))

    def test_smoothed_timesteps_min_dt_zero(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=0)
        timesteps = p.smoothed_timesteps(100, 30, indices)
        assert_equal(timesteps, np.arange(1, 100, 1))

    def test_correct_drift_no_framework(self):
        corrected = parser._correct_drift([], dc)
        assert_equal(len(corrected), 100)
        for i, d in enumerate(corrected):
            assert_equal(d.shape[0], 100)
            assert_equal(d.shape[1], 3)

    def test_correct_drift_framework(self):
        corrected = parser._correct_drift([], dc)
        assert_equal(len(corrected), 100)
        for i, d in enumerate(corrected):
            assert_equal(d.shape[0], 100)
            assert_equal(d.shape[1], 3)

    def test_get_disps(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        dt, disp_3d = p.get_disps(np.arange(20, 100, 1), dc)
        assert_equal(dt, np.arange(20, 100, 1))
        assert_equal(len(disp_3d), 80)
        for i, d in enumerate(disp_3d):
            assert_equal(d.shape[0], 100)
            assert_equal(d.shape[1], 80-i)
            assert_equal(d.shape[2], 3)

    def test_pymatgen_init(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = { 'specie': 'Li',
                      'time_step': 2.0,
                      'step_skip': 50,
                      'min_obs': 50}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 2.0)
        assert_almost_equal(data.step_skip, 50)
        assert_equal(data.indices, list(range(xd.natoms[0])))

    def test_pymatgen_big_timestep(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = { 'specie': 'Li',
                      'time_step': 20.0,
                      'step_skip': 100,
                      'min_obs': 50}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 20.0)
        assert_almost_equal(data.step_skip, 100)
        assert_equal(data.indices, list(range(xd.natoms[0])))

    def test_mda_init(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'), format='LAMMPS')
        da_params = { 'specie': '1',
                      'time_step': 0.005,
                      'step_skip': 250,
                      'min_obs': 50}
        data = parser.MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_equal(data.indices, list(range(204)))

    def test_get_matrix(self):
        matrix = parser._get_matrix([10, 10, 10, 90, 90, 90])
        assert_almost_equal(matrix, np.diag((10, 10, 10)))
