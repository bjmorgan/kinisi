"""
Tests for parser module
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)
# pylint: disable=R0201

import unittest
import numpy as np
import MDAnalysis as mda
from numpy.testing import assert_almost_equal, assert_equal
from pymatgen.io.vasp import Xdatcar
from ase.io import Trajectory
import os
import kinisi
from kinisi import parser

dc = np.random.random(size=(100, 100, 3))
indices = np.arange(0, 100, 1, dtype=int)
time_step = 1.0
step_skip = 1


class TestParser(unittest.TestCase):
    """
    Unit tests for parser module
    """

    def test_parser_init_time_interval(self):
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
        assert_equal(p.delta_t.size, 81)

    def test_parser_disp_3d(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        assert_equal(len(p.disp_3d), 81)
        dt = np.arange(20, 101, 1)
        assert_equal(p.disp_3d[0].shape[0], 100)
        assert_equal(p.disp_3d[0].shape[1], 81)
        assert_equal(p.disp_3d[0].shape[2], 3)

    def test_get_time_intervals(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        time_intervals = p.get_time_intervals(80, 'linear')
        assert_equal(time_intervals, np.linspace(20, 100, 80, dtype=int))

    def test_get_time_intervals_min_dt_zero(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=0)
        time_intervals = p.get_time_intervals(100, 'logarithmic')
        assert_equal(time_intervals, np.unique(np.geomspace(1, 100, 100, dtype=int)))

    def test_correct_drift_no_framework(self):
        corrected = parser.Parser.correct_drift([], dc)
        assert_equal(len(corrected), 100)
        for i, d in enumerate(corrected):
            assert_equal(d.shape[0], 100)
            assert_equal(d.shape[1], 3)

    def test_correct_drift_framework(self):
        corrected = parser.Parser.correct_drift([], dc)
        assert_equal(len(corrected), 100)
        for i, d in enumerate(corrected):
            assert_equal(d.shape[0], 100)
            assert_equal(d.shape[1], 3)

    def test_get_disps(self):
        p = parser.Parser(dc, indices, [], time_step, step_skip, min_dt=20)
        dt, disp_3d, n_samples = p.get_disps(np.arange(20, 101, 1), dc)
        assert_equal(dt, np.arange(20, 101, 1))
        assert_equal(len(disp_3d), 81)
        assert_equal(len(n_samples), 81)
        assert_equal(disp_3d[0].shape[0], 100)
        assert_equal(disp_3d[0].shape[1], 81)
        assert_equal(disp_3d[0].shape[2], 3)

    #Pymatgen tests with VASP XDATCAR files
    def test_pymatgen_init(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0, 'step_skip': 50}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 2.0)
        assert_almost_equal(data.step_skip, 50)
        assert_equal(data.indices, list(range(xd.natoms[0])))

    def test_pymatgen_init_with_indices(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': None, 'time_step': 2.0, 'step_skip': 50, 'specie_indices': [3, 4, 5]}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 2.0)
        assert_almost_equal(data.step_skip, 50)
        assert_equal(data.indices, [3, 4, 5])

    def test_pymatgen_init_with_molecules(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        molecules = [[2, 3, 4], [5, 6, 7]]
        da_params = {'specie': None, 'time_step': 2.0, 'step_skip': 50, 'specie_indices': molecules}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 2.0)
        assert_almost_equal(data.step_skip, 50)
        assert_equal(data.indices, list(range(len(molecules))))

    def test_pymatgen_big_time_interval(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 20.0, 'step_skip': 100}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 20.0)
        assert_almost_equal(data.step_skip, 100)
        assert_equal(data.indices, list(range(xd.natoms[0])))

    def test_pymatgen_init_with_COG(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_center.XDATCAR'))
        da_params = {'specie': None, 'time_step': 1.0, 'step_skip': 1, 'specie_indices': [[1, 2, 3]]}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 1)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, [0])
        assert_almost_equal(data.coords_check, [[[0.2733333, 0.2666667, 0.2      ]]])

    def test_pymatgen_init_with_COM(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_center.XDATCAR'))
        da_params = {
            'specie': None,
            'time_step': 1.0,
            'step_skip': 1,
            'specie_indices': [[1, 2, 3]],
            'masses': [1, 16, 1]
        }
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 1)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, [0])
        assert_almost_equal(data.coords_check, [[[0.3788889, 0.2111111, 0.2      ]]])

    def test_pymatgen_init_with_framwork_indices(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_drift.XDATCAR'))
        da_1_params = {'specie': 'H', 'time_step': 1.0, 'step_skip': 1, 'framework_indices': []}
        data_1 = parser.PymatgenParser(xd.structures, **da_1_params)
        assert_almost_equal(data_1.time_step, 1)
        assert_almost_equal(data_1.step_skip, 1)
        assert_equal(data_1.indices, [0])
        assert_equal(data_1.drift_indices, [])
        assert_equal(data_1.dc[0], np.zeros((4, 3)))
        da_2_params = {'specie': 'H', 'time_step': 1.0, 'step_skip': 1}
        data_2 = parser.PymatgenParser(xd.structures, **da_2_params)
        assert_almost_equal(data_2.time_step, 1)
        assert_almost_equal(data_2.step_skip, 1)
        assert_equal(data_2.indices, [0])
        assert_equal(data_2.drift_indices, list(range(1, 9)))
        print(data_2.dc[0])
        disp_array = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [1, 1, 1]]
        assert_almost_equal(data_2.dc[0], disp_array)

    #ASE tests with ASE traj files
    def test_ase_init(self):
        traj = Trajectory(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase.traj'))
        da_params = {'specie': 'Li', 'time_step': 1e-3, 'step_skip': 1}
        data = parser.ASEParser(traj, **da_params)
        assert_almost_equal(data.time_step, 1e-3)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, list(range(180)))

    def test_ase_init_with_indices(self):
        traj = Trajectory(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase.traj'))
        da_params = {'specie': None, 'time_step': 1e-3, 'step_skip': 1, 'specie_indices': [100, 101, 90]}
        data = parser.ASEParser(traj, **da_params)
        assert_almost_equal(data.time_step, 1e-3)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, [100, 101, 90])

    def test_ase_init_with_molecules(self):
        traj = Trajectory(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase.traj'))
        molecules = [[2, 3, 4], [5, 6, 7]]
        da_params = {'specie': None, 'time_step': 1e-3, 'step_skip': 1, 'specie_indices': molecules}
        data = parser.ASEParser(traj, **da_params)
        assert_almost_equal(data.time_step, 1e-3)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, list(range(len(molecules))))

    def test_ase_init_with_COG(self):
        traj = Trajectory(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase_center.traj'))
        da_params = {'specie': None, 'time_step': 1, 'step_skip': 1, 'specie_indices': [[1, 2, 3]]}
        data = parser.ASEParser(traj, **da_params)
        assert_almost_equal(data.time_step, 1)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, [0])
        assert_almost_equal(data.coords_check, [[[0.2733333, 0.2666667, 0.2]]])

    def test_ase_init_with_COM(self):
        traj = Trajectory(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase_center.traj'))
        da_params = {
            'specie': None,
            'time_step': 1,
            'step_skip': 1,
            'specie_indices': [[1, 2, 3]],
            'masses': [1, 16, 1]
        }
        data = parser.ASEParser(traj, **da_params)
        assert_almost_equal(data.time_step, 1)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, [0])
        assert_almost_equal(data.coords_check, [[[0.3788889, 0.2111111, 0.2]]])

    def test_ase_init_with_framwork_indices(self):
        traj = Trajectory(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_ase_drift.traj'))
        da_1_params = {'specie': 'H', 'time_step': 1, 'step_skip': 1, 'framework_indices': []}
        data_1 = parser.ASEParser(traj, **da_1_params)
        assert_almost_equal(data_1.time_step, 1)
        assert_almost_equal(data_1.step_skip, 1)
        assert_equal(data_1.indices, [0])
        assert_equal(data_1.drift_indices, [])
        assert_equal(data_1.dc[0], np.zeros((4, 3)))
        da_2_params = {'specie': 'H', 'time_step': 1, 'step_skip': 1}
        data_2 = parser.ASEParser(traj, **da_2_params)
        assert_almost_equal(data_2.time_step, 1)
        assert_almost_equal(data_2.step_skip, 1)
        assert_equal(data_2.indices, [0])
        assert_equal(data_2.drift_indices, list(range(1, 9)))
        print(data_2.dc[0])
        disp_array = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [1, 1, 1]]
        assert_almost_equal(data_2.dc[0], disp_array)

    #MDAnalysis tests with LAMMPS data and dump files
    def test_mda_init(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                          format='LAMMPS')
        da_params = {'specie': '1', 'time_step': 0.005, 'step_skip': 250}
        data = parser.MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_equal(data.indices, list(range(204)))

    def test_mda_init_with_indices(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                          format='LAMMPS')
        da_params = {'specie': None, 'time_step': 0.005, 'step_skip': 250, 'specie_indices': [208, 212]}
        data = parser.MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_equal(data.indices, [208, 212])

    def test_mda_init_with_molecules(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                          format='LAMMPS')
        molecules = [[2, 3, 4], [5, 6, 7]]
        da_params = {'specie': None, 'time_step': 0.005, 'step_skip': 250, 'specie_indices': molecules}
        data = parser.MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 0.005)
        assert_almost_equal(data.step_skip, 250)
        assert_almost_equal(data.coords_check[0, 0], [0.5169497, 0.1174514, 0.3637794])
        assert_equal(data.indices, list(range(len(molecules))))

    def test_mda_init_with_COG(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.traj'),
                          topology_format='DATA',
                          format='LAMMPSDUMP')
        da_params = {'specie': None, 'time_step': 1, 'step_skip': 1, 'specie_indices': [[1, 2, 3]]}
        data = parser.MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 1)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, [0])
        assert_almost_equal(data.coords_check, [[[0.2733333, 0.2666667, 0.1999999]]])

    def test_mda_init_with_COM(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_center.traj'),
                          topology_format='DATA',
                          format='LAMMPSDUMP')
        da_params = {
            'specie': None,
            'time_step': 1,
            'step_skip': 1,
            'specie_indices': [[1, 2, 3]],
            'masses': [1, 16, 1]
        }
        data = parser.MDAnalysisParser(xd, **da_params)
        assert_almost_equal(data.time_step, 1)
        assert_almost_equal(data.step_skip, 1)
        assert_equal(data.indices, [0])
        assert_almost_equal(data.coords_check, [[[0.3788889, 0.2111111, 0.2]]])

    def test_mda_init_with_framwork_indices(self):
        xd = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_drift.data'),
                          os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS_drift.traj'),
                          topology_format='DATA',
                          format='LAMMPSDUMP')
        da_1_params = {'specie': '1', 'time_step': 1, 'step_skip': 1, 'framework_indices': []}
        data_1 = parser.MDAnalysisParser(xd, **da_1_params)
        assert_almost_equal(data_1.time_step, 1)
        assert_almost_equal(data_1.step_skip, 1)
        assert_equal(data_1.indices, [0])
        assert_equal(data_1.drift_indices, [])
        assert_equal(data_1.dc[0], np.zeros((4, 3)))
        da_2_params = {'specie': '1', 'time_step': 1, 'step_skip': 1}
        data_2 = parser.MDAnalysisParser(xd, **da_2_params)
        assert_almost_equal(data_2.time_step, 1)
        assert_almost_equal(data_2.step_skip, 1)
        assert_equal(data_2.indices, [0])
        assert_equal(data_2.drift_indices, list(range(1, 9)))
        print(data_2.dc[0])
        disp_array = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [1, 1, 1]]
        assert_almost_equal(data_2.dc[0], disp_array, decimal=6)
