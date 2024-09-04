"""
Parser for kinisi. This module is responsible for reading in input files from :py:mod:`pymatgen`,
:py:mod:`MDAnalysis`, and :py:mod:`ase`.
"""

# Copyright (c) kinisi developers. 
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

from typing import List, Tuple, Union
import numpy as np
import scipp as sc
from tqdm import tqdm


DIMENSIONALITY = {
    'x': np.s_[0],
    'y': np.s_[1],
    'z': np.s_[2],
    'xy': np.s_[:2],
    'xz': np.s_[::2],
    'yz': np.s_[1:],
    'xyz': np.s_[:],
    b'x': np.s_[0],
    b'y': np.s_[1],
    b'z': np.s_[2],
    b'xy': np.s_[:2],
    b'xz': np.s_[::2],
    b'yz': np.s_[1:],
    b'xyz': np.s_[:]
}


class Parser:
    """
    Superclass for parsers
    """
    def __init__(self,
                 coords: sc.Variable,
                 lattice: sc.Variable,
                 indices: sc.Variable,
                 drift_indices: sc.Variable,
                 time_step: sc.Variable,
                 step_skip: sc.Variable,
                 dt: sc.Variable = None,
                 dimension: str = 'xyz'):
        self.time_step = time_step
        self.step_skip = step_skip
        self.indices = indices
        self.drift_indices = drift_indices
        self._dimension = dimension
        self._volume = None
        self.dt = dt
        if self.dt is None:
            self.dt_int = sc.arange(start=1, stop=coords.sizes['time'], step=1, dim='timestep')
            self.dt = self.dt_int * time_step * step_skip

        disp = self.calculate_displacements(coords, lattice)
        drift_corrected = self.correct_drift(disp)

        self._slice = DIMENSIONALITY[dimension.lower()]
        drift_corrected = drift_corrected['dimension', self._slice]
        self.dimensionality = drift_corrected.sizes['dimension'] * sc.units.dimensionless

        self.displacements = drift_corrected['atom', indices]
    
    def calculate_displacements(self, 
                                coords: sc.Variable,
                                lattice: sc.Variable) -> sc.Variable:
        lattice_inv = np.linalg.inv(lattice.values)
        wrapped = sc.array(dims=coords.dims, values=np.einsum('jik,jkl->jil', coords.values, lattice.values), unit=coords.unit)
        wrapped_diff = sc.array(dims=['obs'] + list(coords.dims[1:]), values=(wrapped['time', 1:] - wrapped['time', :-1]).values, unit=coords.unit)
        diff_diff = sc.array(dims=wrapped_diff.dims, values=np.einsum('jik,jkl->jil', np.floor(np.einsum('jik,jkl->jil', wrapped_diff.values, lattice_inv[1:]) + 0.5), lattice.values[1:]), unit=coords.unit)
        unwrapped_diff = wrapped_diff - diff_diff
        return sc.cumsum(unwrapped_diff, 'obs')
    
    def correct_drift(self, disp: sc.Variable) -> sc.Variable:
        return disp - sc.mean(disp['atom', self.drift_indices.values], 'atom')
    

class PymatgenParser(Parser):
    """
    Parser for pymatgen structures.
    """
    def __init__(self, 
                 structures: List['pymatgen.core.structure.Structure'],
                 specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
                 time_step: sc.Variable,
                 step_skip: sc.Variable,
                 dt: sc.Variable = None,
                 dimension: str = 'xyz',
                 progress: bool = True,
                 distance_unit: sc.Unit = sc.units.angstrom):
        self.distance_unit = distance_unit

        structure, coords, latt = self.get_structure_coords_latt(structures, progress)
        indices, drift_indices = self.get_indices(structure, specie)

        super().__init__(coords, latt, indices, drift_indices, time_step, step_skip, dt, dimension)
        self._volume = structure.volume * self.distance_unit ** 3
        
    def get_structure_coords_latt(self, 
                                  structures: List['pymatgen.core.structure.Structure'], 
                                  progress: bool = True) -> Tuple["pymatgen.core.structure.Structure", sc.Variable, sc.Variable]:
        first = True
        coords_l = []
        latt_l = []
        if progress:
            iterator = tqdm(structures, desc='Reading Trajectory')
        else:
            iterator = structures
        for struct in iterator:
            if first:
                structure = struct
                first = False
            coords_l.append(np.array(struct.frac_coords))
            latt_l.append(np.array(struct.lattice.matrix))

        coords_l.insert(0, coords_l[0])
        latt_l.insert(0, latt_l[0])
        coords_l = np.array(coords_l)
        latt_l = np.array(latt_l)
        coords = sc.array(dims=['time', 'atom', 'dimension'], values=coords_l, unit=self.distance_unit)
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt_l, unit=self.distance_unit)
        return structure, coords, latt
    
    def get_indices(self, 
                    structure: 'pymatgen.core.structure.Structure',
                    specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie']) -> Tuple[sc.Variable, sc.Variable]:   
        indices = []
        drift_indices = []
        for i, site in enumerate(structure):
            if site.specie.__str__() in specie:
                indices.append(i)
            else:
                drift_indices.append(i)
        indices = sc.Variable(dims=['atom'], values=indices)
        drift_indices = sc.Variable(dims=['atom'], values=drift_indices)
        return indices, drift_indices
