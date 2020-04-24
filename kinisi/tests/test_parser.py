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
from numpy.testing import assert_almost_equal, assert_equal
from pymatgen.core import Structure
from pymatgen.io.vasp import Xdatcar
import os
import kinisi
from kinisi import parser
from uravu.distribution import Distribution
from uravu.utils import straight_line


class TestParser(unittest.TestCase):
    """
    Unit tests for parser module
    """

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
        assert_equal(1, 1)

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
        assert_equal(1, 1)