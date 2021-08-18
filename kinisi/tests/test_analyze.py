"""
Tests for analyze module

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

import os
import warnings
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from pymatgen.io.vasp import Xdatcar
import MDAnalysis as mda
from uravu.distribution import Distribution
import kinisi
from kinisi.analyze import Analyzer, DiffusionAnalyzer, JumpDiffusionAnalyzer, ConductivityAnalyzer, _flatten_list

file_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz')
xd = Xdatcar(file_path)
da_params = {'specie': 'Li',
             'time_step': 2.0, 
             'step_skip': 50,
             'min_obs': 1}
md = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                  os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'), format='LAMMPS')
db_params = {'specie': '1',
             'time_step': 0.005,
             'step_skip': 250,
             'min_obs': 1}


class TestAnalyzer(unittest.TestCase):
    """
    Tests for the Analyzer base class.
    """
    def test_structure_pmg(self):
        a = Analyzer(xd.structures, parser_params=da_params, dtype='vasp')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13.9)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (192, 139, 3)
        assert a.disp_3d[-1].shape == (192, 1, 3)
    
    def test_xdatcar_pmg(self):
        a = Analyzer(xd, parser_params=da_params, dtype='vasp')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13.9)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (192, 139, 3)
        assert a.disp_3d[-1].shape == (192, 1, 3)
    
    def test_file_path_pmg(self):
        a = Analyzer(file_path, parser_params=da_params, dtype='vasp')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13.9)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (192, 139, 3)
        assert a.disp_3d[-1].shape == (192, 1, 3)

    def test_identical_structure_pmg(self):
        a = Analyzer([xd.structures, xd.structures], parser_params=da_params, dtype='identicalvasp')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13.9)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (384, 139, 3)
        assert a.disp_3d[-1].shape == (384, 1, 3)
    
    def test_identical_xdatcar_pmg(self):
        a = Analyzer([xd, xd], parser_params=da_params, dtype='identicalvasp')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13.9)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (384, 139, 3)
        assert a.disp_3d[-1].shape == (384, 1, 3)

    def test_identical_file_path_pmg(self):
        a = Analyzer([file_path, file_path], parser_params=da_params, dtype='identicalvasp')
        assert a.delta_t.size == 139
        assert a.delta_t.max() == 13.90
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (384, 139, 3)
        assert a.disp_3d[-1].shape == (384, 1, 3)

    def test_consecutive_structure_pmg(self):
        a = Analyzer([xd.structures, xd.structures], parser_params=da_params, dtype='consecutivevasp')
        assert a.delta_t.size == 93
        assert_almost_equal(a.delta_t.max(), 27.7)
        assert len(a.disp_3d) == 93
        assert a.disp_3d[0].shape == (192, 279, 3)
        assert a.disp_3d[-1].shape == (192, 3, 3)
    
    def test_consecutive_xdatcar_pmg(self):
        a = Analyzer([xd, xd], parser_params=da_params, dtype='consecutivevasp')
        assert a.delta_t.size == 93
        assert_almost_equal(a.delta_t.max(), 27.7)
        assert len(a.disp_3d) == 93
        assert a.disp_3d[0].shape == (192, 279, 3)
        assert a.disp_3d[-1].shape == (192, 3, 3)

    def test_consecutive_file_path_pmg(self):
        a = Analyzer([file_path, file_path], parser_params=da_params, dtype='consecutivevasp')
        assert a.delta_t.size == 93
        assert_almost_equal(a.delta_t.max(), 27.7)
        assert len(a.disp_3d) == 93
        assert a.disp_3d[0].shape == (192, 279, 3)
        assert a.disp_3d[-1].shape == (192, 3, 3)

    def test_pymatgen_bad_input(self):
        with self.assertRaises(ValueError):
            a = Analyzer(1, parser_params=da_params, dtype='vasp')

    def test_mdauniverse(self):
        a = Analyzer(md, parser_params=db_params, dtype='mdanalysis')
        assert a.delta_t.size == 120
        assert_almost_equal(a.delta_t.max(), 248.75)
        assert len(a.disp_3d) == 120
        assert a.disp_3d[0].shape == (204, 120, 3)
        assert a.disp_3d[-1].shape == (204, 1, 3)
    
    def test_list_bad_input(self):
        with self.assertRaises(ValueError):
            a = Analyzer([file_path, file_path], parser_params=da_params, dtype='consecutievasp')
    
    def test_list_bad_mda(self):
        with self.assertRaises(ValueError):
            a = Analyzer(file_path, parser_params=db_params, dtype='mdanalysis')


class TestDiffusionAnalyzer(unittest.TestCase):
    """
    Tests for the DiffusionAnalyzer base class.
    """
    def test_properties(self):
        with warnings.catch_warnings(record=True) as w:
            a = DiffusionAnalyzer(xd.structures, parser_params=da_params, bootstrap_params={'random_state': np.random.RandomState(0)}, dtype='vasp')
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.msd, a._diff.n)
            assert_almost_equal(a.msd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert a.D == None
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as w:
            a = DiffusionAnalyzer(xd.structures, parser_params=da_params, bootstrap_params={'random_state': np.random.RandomState(0)}, dtype='vasp')
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.msd, a._diff.n)
            assert_almost_equal(a.msd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.diffusion()
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.D, Distribution)
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)


class TestJumpDiffusionAnalyzer(unittest.TestCase):
    """
    Tests for the JumpDiffusionAnalyzer base class.
    """
    def test_properties(self):
        with warnings.catch_warnings(record=True) as w:
            a = JumpDiffusionAnalyzer(xd.structures, parser_params=da_params, bootstrap_params={'random_state': np.random.RandomState(0)}, dtype='vasp')
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.tmsd, a._diff.n)
            assert_almost_equal(a.tmsd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as w:
            a = JumpDiffusionAnalyzer(xd.structures, parser_params=da_params, bootstrap_params={'random_state': np.random.RandomState(0)}, dtype='vasp')
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.tmsd, a._diff.n)
            assert_almost_equal(a.tmsd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.jump_diffusion()
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.D_J, Distribution)
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)


class TestConductivityAnalyzer(unittest.TestCase):
    """
    Tests for the ConductivityAnalyzer base class.
    """
    def test_properties(self):
        with warnings.catch_warnings(record=True) as w:
            a = ConductivityAnalyzer(xd.structures, 1, parser_params=da_params, bootstrap_params={'random_state': np.random.RandomState(0)}, dtype='vasp')
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mscd, a._diff.n)
            assert_almost_equal(a.mscd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as w:
            a = ConductivityAnalyzer(xd.structures, 1, parser_params=da_params, bootstrap_params={'random_state': np.random.RandomState(0)}, dtype='vasp')
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mscd, a._diff.n)
            assert_almost_equal(a.mscd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.conductivity(100)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.sigma, Distribution)
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)
    

class TestFunctions(unittest.TestCase):
    """
    Tests for other functions
    """
    def test__flatten_list(self):
        a_list = [[1, 2, 3], [4, 5]]
        result = _flatten_list(a_list)
        assert result == [1, 2, 3, 4, 5]