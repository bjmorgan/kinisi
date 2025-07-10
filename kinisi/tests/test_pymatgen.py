import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import scipp as sc
from pymatgen.io.vasp import Xdatcar
import os

import kinisi
from kinisi import parser
from kinisi.pymatgen import PymatgenParser

class TestPymatgenParser(unittest.TestCase):
    """
    Unit tests for the pymatgen module
    """

    def test_pymatgen_init(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0, 'step_skip': 50}
        data = parser.PymatgenParser(xd.structures, **da_params)
        assert_almost_equal(data.time_step, 2.0)
        assert_almost_equal(data.step_skip, 50)
        assert_equal(data.indices, list(range(xd.natoms[0])))

    def test_pymatgen_datagroup_round_trip(self):
        xd = Xdatcar(os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz'))
        da_params = {'specie': 'Li', 'time_step': 2.0 * sc.Unit('fs'), 'step_skip': 50 * sc.Unit('dimensionless')}
        data = PymatgenParser(xd.structures, **da_params)
        datagroup = data._to_datagroup()
        data_2 = parser.Parser._from_datagroup(datagroup)
        assert_equal(vars(data), vars(data_2))
        assert_equal(type(data), type(data_2))
        data_3 = PymatgenParser._from_datagroup(datagroup)
        assert_equal(vars(data), vars(data_3))
        assert_equal(type(data), type(data_3))