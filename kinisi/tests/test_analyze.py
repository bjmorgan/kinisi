"""
Tests for parser module

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
import kinisi
from kinisi.analyze import Analyzer, MSDAnalyzer

file_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz')
xd = Xdatcar(file_path)
da_params = {'specie': 'Li',
             'time_step': 2.0, 
             'step_skip': 50,
             'min_obs': 1}


class TestAnalyzer(unittest.TestCase):
    """
    Tests for the Analyzer base class.
    """
    def test_structure(self):
        a = Analyzer(xd.structures, parser_params=da_params, dtype='pymatgen')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13900.)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (192, 139, 3)
        assert a.disp_3d[-1].shape == (192, 1, 3)
    
    def test_xdatcar(self):
        a = Analyzer(xd, parser_params=da_params, dtype='pymatgen')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13900.)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (192, 139, 3)
        assert a.disp_3d[-1].shape == (192, 1, 3)
    
    def test_file_path(self):
        a = Analyzer(file_path, parser_params=da_params, dtype='pymatgen')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13900.)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (192, 139, 3)
        assert a.disp_3d[-1].shape == (192, 1, 3)

    def test_identical_structure(self):
        a = Analyzer([xd.structures, xd.structures], parser_params=da_params, dtype='identicalpymatgen')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13900.)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (384, 139, 3)
        assert a.disp_3d[-1].shape == (384, 1, 3)
    
    def test_identical_xdatcar(self):
        a = Analyzer([xd, xd], parser_params=da_params, dtype='identicalpymatgen')
        assert a.delta_t.size == 139
        assert_almost_equal(a.delta_t.max(), 13900.)
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (384, 139, 3)
        assert a.disp_3d[-1].shape == (384, 1, 3)

    def test_identical_file_path(self):
        a = Analyzer([file_path, file_path], parser_params=da_params, dtype='identicalpymatgen')
        assert a.delta_t.size == 139
        assert a.delta_t.max() == 13900.
        assert len(a.disp_3d) == 139
        assert a.disp_3d[0].shape == (384, 139, 3)
        assert a.disp_3d[-1].shape == (384, 1, 3)

    def test_consecutive_structure(self):
        a = Analyzer([xd.structures, xd.structures], parser_params=da_params, dtype='consecutivepymatgen')
        assert a.delta_t.size == 93
        assert_almost_equal(a.delta_t.max(), 27700.)
        assert len(a.disp_3d) == 93
        assert a.disp_3d[0].shape == (192, 279, 3)
        assert a.disp_3d[-1].shape == (192, 3, 3)
    
    def test_consecutive_xdatcar(self):
        a = Analyzer([xd, xd], parser_params=da_params, dtype='consecutivepymatgen')
        assert a.delta_t.size == 93
        assert_almost_equal(a.delta_t.max(), 27700.)
        assert len(a.disp_3d) == 93
        assert a.disp_3d[0].shape == (192, 279, 3)
        assert a.disp_3d[-1].shape == (192, 3, 3)

    def test_consecutive_file_path(self):
        a = Analyzer([file_path, file_path], parser_params=da_params, dtype='consecutivepymatgen')
        assert a.delta_t.size == 93
        assert_almost_equal(a.delta_t.max(), 27700.)
        assert len(a.disp_3d) == 93
        assert a.disp_3d[0].shape == (192, 279, 3)
        assert a.disp_3d[-1].shape == (192, 3, 3)

    def test_pymatgen_bad_input(self):
        with self.assertRaises(ValueError):
            a = Analyzer(1, parser_params=da_params, dtype='pymatgen')
    
    def test_list_bad_input(self):
        with self.assertRaises(ValueError):
            a = Analyzer([file_path, file_path], parser_params=da_params, dtype='consecutiepymatgen')


class TestMSDAnalyzer(unittest.TestCase):
    """
    Tests for the MSDAnalyzer base class.
    """
    def test_properties(self):
        with warnings.catch_warnings(record=True) as w:
            a = MSDAnalyzer(xd.structures, parser_params=da_params, bootstrap_params={'random_state': np.random.RandomState(0)}, dtype='pymatgen')
            assert_almost_equal(a.dt, a._diff.dt)
            assert_almost_equal(a.msd, a._diff.msd)
            assert_almost_equal(a.msd_std, a._diff.msd_std)
            for i in range(len(a.dr)):
                assert_almost_equal(a.dr[i].samples, a._diff.euclidian_displacements[i].samples)
            assert a.ngp_max == a._diff.dt[a._diff.ngp.argmax()]
            assert a.D == None
            assert a.D_offset == None
            assert issubclass(w[0].category, UserWarning)
            assert "maximum" in str(w[0].message)
    