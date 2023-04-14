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
from numpy.testing import assert_almost_equal, assert_equal
from pymatgen.io.vasp import Xdatcar
import MDAnalysis as mda
from uravu.distribution import Distribution
import kinisi
from kinisi.analyze import DiffusionAnalyzer, JumpDiffusionAnalyzer, ConductivityAnalyzer
from kinisi.analyzer import Analyzer, _flatten_list

file_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz')
xd = Xdatcar(file_path)
da_params = {'specie': 'Li', 'time_step': 2.0, 'step_skip': 50}
md = mda.Universe(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.data'),
                  os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_LAMMPS.dcd'),
                  format='LAMMPS')
db_params = {'specie': '1', 'time_step': 0.005, 'step_skip': 250}


class TestAnalyzer(unittest.TestCase):
    """
    Tests for the Analyzer base class.
    """

    def test_dict_roundtrip(self):
        a = Analyzer._from_pymatgen(xd.structures, parser_params=da_params)
        b = Analyzer.from_dict(a.to_dict())
        assert_equal(a._delta_t, b._delta_t)
        for i, d in enumerate(a._disp_3d):
            assert_equal(d, b._disp_3d[i])
        assert a._volume == b._volume

    def test_structure_pmg(self):
        a = Analyzer._from_pymatgen(xd.structures, parser_params=da_params)
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 14.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (192, 140, 3)
        assert a._disp_3d[-1].shape == (192, 1, 3)

    def test_xdatcar_pmg(self):
        a = Analyzer._from_Xdatcar(xd, parser_params=da_params)
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 14.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (192, 140, 3)
        assert a._disp_3d[1].shape == (192, 139, 3)
        assert a._disp_3d[-1].shape == (192, 1, 3)

    def test_file_path_pmg(self):
        a = Analyzer._from_file(file_path, parser_params=da_params)
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 14.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (192, 140, 3)
        assert a._disp_3d[-1].shape == (192, 1, 3)

    def test_identical_structure_pmg(self):
        a = Analyzer._from_pymatgen([xd.structures, xd.structures], parser_params=da_params, dtype='identical')
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 14.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (384, 140, 3)
        assert a._disp_3d[-1].shape == (384, 1, 3)

    def test_identical_xdatcar_pmg(self):
        a = Analyzer._from_Xdatcar([xd, xd], parser_params=da_params, dtype='identical')
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 14.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (384, 140, 3)
        assert a._disp_3d[-1].shape == (384, 1, 3)

    def test_identical_file_path_pmg(self):
        a = Analyzer._from_file([file_path, file_path], parser_params=da_params, dtype='identical')
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 14.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (384, 140, 3)
        assert a._disp_3d[-1].shape == (384, 1, 3)

    def test_consecutive_structure_pmg(self):
        a = Analyzer._from_pymatgen([xd.structures, xd.structures], parser_params=da_params, dtype='consecutive')
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 28.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (192, 280, 3)
        assert a._disp_3d[-1].shape == (192, 1, 3)

    def test_consecutive_xdatcar_pmg(self):
        a = Analyzer._from_Xdatcar([xd, xd], parser_params=da_params, dtype='consecutive')
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 28.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (192, 280, 3)
        assert a._disp_3d[-1].shape == (192, 1, 3)

    def test_consecutive_file_path_pmg(self):
        a = Analyzer._from_file([file_path, file_path], parser_params=da_params, dtype='consecutive')
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 28.0)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (192, 280, 3)
        assert a._disp_3d[-1].shape == (192, 1, 3)

    def test_mdauniverse(self):
        a = Analyzer._from_universe(md, parser_params=db_params)
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 250.)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (204, 200, 3)
        assert a._disp_3d[-1].shape == (204, 1, 3)

    def test_identical_mdauniverse(self):
        a = Analyzer._from_universe([md, md], parser_params=db_params, dtype='identical')
        assert a._delta_t.size == 100
        assert_almost_equal(a._delta_t.max(), 250.)
        assert len(a._disp_3d) == 100
        assert a._disp_3d[0].shape == (408, 200, 3)
        assert a._disp_3d[-1].shape == (408, 1, 3)

    def test_list_bad_input(self):
        with self.assertRaises(ValueError):
            _ = Analyzer._from_file([file_path, file_path], parser_params=da_params, dtype='consecutie')

    def test_list_bad_mda(self):
        with self.assertRaises(ValueError):
            _ = Analyzer._from_universe(file_path, parser_params=db_params, dtype='consecutie')


class TestDiffusionAnalyzer(unittest.TestCase):
    """
    Tests for the DiffusionAnalyzer base class.
    """

    def test_properties(self):
        a = DiffusionAnalyzer.from_pymatgen(xd.structures,
                                            parser_params=da_params,
                                            bootstrap_params={'random_state': np.random.RandomState(0)})
        assert_almost_equal(a.dt, a._diff.dt)
        assert_almost_equal(a.msd, a._diff.n)
        assert_almost_equal(a.msd_std, a._diff.s)
        for i in range(len(a.dr)):
            assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
        assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
        assert a.D is None

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as _:
            a = DiffusionAnalyzer.from_pymatgen(xd.structures,
                                                parser_params=da_params,
                                                bootstrap_params={'random_state': np.random.RandomState(0)})
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.msd, a._diff.n)
            assert_almost_equal(a.msd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.diffusion()
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.D, Distribution)
            assert a.flatchain.shape == (3200, 2)

    def test_dictionary_roundtrip(self):
        with warnings.catch_warnings(record=True) as _:
            a = DiffusionAnalyzer.from_pymatgen(xd.structures,
                                                parser_params=da_params,
                                                bootstrap_params={'random_state': np.random.RandomState(0)})
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.msd, a._diff.n)
            assert_almost_equal(a.msd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.diffusion()
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.D, Distribution)
            assert a.flatchain.shape == (3200, 2)
            b = DiffusionAnalyzer.from_dict(a.to_dict())
            assert_equal(a.dt, b.dt)
            assert_equal(a.msd, b.msd)
            assert_equal(a.msd_std, b.msd_std)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, b.dr[i].samples)
            assert a.ngp_max == b.ngp_max
            assert_equal(a.D.samples, b.D.samples)
            assert_equal(a.flatchain, b.flatchain)


class TestJumpDiffusionAnalyzer(unittest.TestCase):
    """
    Tests for the JumpDiffusionAnalyzer base class.
    """

    def test_properties(self):
        with warnings.catch_warnings(record=True) as w:
            a = JumpDiffusionAnalyzer.from_pymatgen(xd.structures,
                                                    parser_params=da_params,
                                                    bootstrap_params={'random_state': np.random.RandomState(0)})
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mstd, a._diff.n)
            assert_almost_equal(a.mstd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as w:
            a = JumpDiffusionAnalyzer.from_pymatgen(xd.structures,
                                                    parser_params=da_params,
                                                    bootstrap_params={'random_state': np.random.RandomState(0)})
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mstd, a._diff.n)
            assert_almost_equal(a.mstd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.jump_diffusion()
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.D_J, Distribution)
            assert a.flatchain.shape == (3200, 2)

    def test_dictionary_roundtrip(self):
        with warnings.catch_warnings(record=True) as w:
            a = JumpDiffusionAnalyzer.from_pymatgen(xd.structures,
                                                    parser_params=da_params,
                                                    bootstrap_params={'random_state': np.random.RandomState(0)})
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mstd, a._diff.n)
            assert_almost_equal(a.mstd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.jump_diffusion()
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.D_J, Distribution)
            assert a.flatchain.shape == (3200, 2)
            b = JumpDiffusionAnalyzer.from_dict(a.to_dict())
            assert_equal(a.dt, b.dt)
            assert_equal(a.mstd, b.mstd)
            assert_equal(a.mstd_std, b.mstd_std)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, b.dr[i].samples)
            assert a.ngp_max == b.ngp_max
            assert_equal(a.D_J.samples, b.D_J.samples)
            assert_equal(a.flatchain, b.flatchain)


class TestConductivityAnalyzer(unittest.TestCase):
    """
    Tests for the ConductivityAnalyzer base class.
    """

    def test_properties(self):
        with warnings.catch_warnings(record=True) as w:
            a = ConductivityAnalyzer.from_pymatgen(xd.structures,
                                                   parser_params=da_params,
                                                   bootstrap_params={'random_state': np.random.RandomState(0)},
                                                   ionic_charge=1)
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mscd, a._diff.n)
            assert_almost_equal(a.mscd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]

    def test_diffusion(self):
        with warnings.catch_warnings(record=True) as w:
            a = ConductivityAnalyzer.from_pymatgen(xd.structures,
                                                   parser_params=da_params,
                                                   bootstrap_params={'random_state': np.random.RandomState(0)},
                                                   ionic_charge=1)
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mscd, a._diff.n)
            assert_almost_equal(a.mscd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.conductivity(100)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.sigma, Distribution)
            assert a.flatchain.shape == (3200, 2)

    def test_dictionary_roundtrip(self):
        with warnings.catch_warnings(record=True) as w:
            a = ConductivityAnalyzer.from_pymatgen(xd.structures,
                                                   parser_params=da_params,
                                                   bootstrap_params={'random_state': np.random.RandomState(0)})
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.mscd, a._diff.n)
            assert_almost_equal(a.mscd_std, a._diff.s)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            a.conductivity(100)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert isinstance(a.sigma, Distribution)
            assert a.flatchain.shape == (3200, 2)
            b = ConductivityAnalyzer.from_dict(a.to_dict())
            assert_equal(a.dt, b.dt)
            assert_equal(a.mscd, b.mscd)
            assert_equal(a.mscd_std, b.mscd_std)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, b.dr[i].samples)
            assert a.ngp_max == b.ngp_max
            assert_equal(a.sigma.samples, b.sigma.samples)
            assert_equal(a.flatchain, b.flatchain)


class TestFunctions(unittest.TestCase):
    """
    Tests for other functions
    """

    def test__flatten_list(self):
        a_list = [[1, 2, 3], [4, 5]]
        result = _flatten_list(a_list)
        assert result == [1, 2, 3, 4, 5]
