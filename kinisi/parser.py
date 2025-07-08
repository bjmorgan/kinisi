"""
Parsers for kinisi. This module is responsible for reading in input files from :py:mod:`pymatgen`,
:py:mod:`MDAnalysis`, and :py:mod:`ase`.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61) and Harry Richardson (Harry-Rich).

from typing import List, Tuple, Union
import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType
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
    The base class for object parsing. 

    This class takes coordinates, lattice parameters, and indices to give the appropriate displacements back.

    :param coords: The fractional coordiates of the atoms in the trajectory. This should be a :py:mod:`scipp`
        array type object with dimensions of 'atom', 'time', and 'dimension'.
    :param lattice: A series of matrices that describe the lattice in each step in the trajectory.
        A :py:mod:`scipp` array with dimensions of 'time', 'dimension1', and 'dimension2'.
    :param indices: Indices for the atoms in the trajectory used in the diffusion calculation.
    :param drift_indices: Indices for the atoms in the trajectory that should not be used in the diffusion
    :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note, 
        that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit 
        that is use for the time interval values.
    :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
        output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
        a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
    :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
        ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with 
        a step size the same as the smallest interval.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    """

    def __init__(self,
                 coords: VariableLikeType,
                 lattice: VariableLikeType,
                 indices: VariableLikeType,
                 drift_indices: VariableLikeType,
                 time_step: VariableLikeType,
                 step_skip: VariableLikeType,
                 dt: VariableLikeType = None,
                 dimension: str = 'xyz'):
        self.time_step = time_step
        self.step_skip = step_skip
        self.indices = indices
        self.drift_indices = drift_indices
        self._dimension = dimension
        self._volume = None
        self.dt = dt
        if self.dt is None:
            self.dt_int = sc.arange(start=1, stop=coords.sizes['time'], step=1, dim='time interval')
            self.dt = self.dt_int * time_step * step_skip
        self.dt_int = (self.dt / (time_step * step_skip)).astype(int)

        disp = self.calculate_displacements(coords, lattice)
        drift_corrected = self.correct_drift(disp)

        self._slice = DIMENSIONALITY[dimension.lower()]
        drift_corrected = drift_corrected['dimension', self._slice]
        self.dimensionality = drift_corrected.sizes['dimension'] * sc.units.dimensionless

        self.displacements = drift_corrected['atom', indices]

    def calculate_displacements(self, coords: VariableLikeType, lattice: VariableLikeType) -> VariableLikeType:
        """
        Calculate the absolute displacements of the atoms in the trajectory.
        
        :param coords: The fractional coordiates of the atoms in the trajectory. This should be a :py:mod:`scipp`
            array type object with dimensions of 'atom', 'time', and 'dimension'.
        :param lattice: A series of matrices that describe the lattice in each step in the trajectory.
            A :py:mod:`scipp` array with dimensions of 'time', 'dimension1', and 'dimension2'.
            
        :return: The absolute displacements of the atoms in the trajectory.
        """
        lattice_inv = np.linalg.inv(lattice.values)
        wrapped = sc.array(dims=coords.dims,
                           values=np.einsum('jik,jkl->jil', coords.values, lattice.values),
                           unit=coords.unit)
        wrapped_diff = sc.array(dims=['obs'] + list(coords.dims[1:]),
                                values=(wrapped['time', 1:] - wrapped['time', :-1]).values,
                                unit=coords.unit)
        diff_diff = sc.array(dims=wrapped_diff.dims,
                             values=np.einsum(
                                 'jik,jkl->jil',
                                 np.floor(np.einsum('jik,jkl->jil', wrapped_diff.values, lattice_inv[1:]) + 0.5),
                                 lattice.values[1:]),
                             unit=coords.unit)
        unwrapped_diff = wrapped_diff - diff_diff
        return sc.cumsum(unwrapped_diff, 'obs')

    def correct_drift(self, disp: VariableLikeType) -> VariableLikeType:
        """
        Perform drift correction, such that the displacement is calculated normalised to any framework drift.

        :param disp: Displacements for all atoms in the simulation. A :py:mod:`scipp` array with dimensions
            of `obs`, `atom` and `dimension`. 

        :return: Displacements corrected to account for drift of a framework.
        """
        return disp - sc.mean(disp['atom', self.drift_indices.values], 'atom')


class PymatgenParser(Parser):
    """
    Parser for pymatgen structures.

    This takes a list of pymatgen structures as an input. 

    :param structures: Structures ordered in sequence of run.
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note, 
        that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit 
        that is use for the time interval values.
    :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
        output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
        a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
    :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
        ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with 
        a step size the same as the smallest interval.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param distance_unit: The unit of distance used in the input structures. Optional, defaults to angstroms.
    :param progress: Whether to show a progress bar when reading in the structures. Optional, defaults to `True`.
    """

    def __init__(
        self,
        structures: List['pymatgen.core.structure.Structure'],
        specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
        time_step: VariableLikeType,
        step_skip: VariableLikeType,
        dt: VariableLikeType = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: VariableLikeType = None,
        masses: VariableLikeType = None,
        progress: bool = True,
    ):
        self.distance_unit = distance_unit

        structure, coords, latt = self.get_structure_coords_latt(structures, progress)

        if specie is not None:
            indices, drift_indices = self.get_indices(structure, specie)
        elif isinstance(specie_indices, sc.Variable):
            if len(specie_indices.dims) > 1:
                coords, indices, drift_indices = _get_molecules(structure, coords, specie_indices, masses,
                                                                distance_unit)
            else:
                indices, drift_indices = _get_framework(structure, specie_indices)
        else:
            raise TypeError('Unrecognized type for specie or specie_indices, specie_indices must be a sc.array')

        super().__init__(coords, latt, indices, drift_indices, time_step, step_skip, dt, dimension)
        self._volume = structure.volume * self.distance_unit**3

    def get_structure_coords_latt(
            self,
            structures: List['pymatgen.core.structure.Structure'],
            progress: bool = True) -> Tuple["pymatgen.core.structure.Structure", VariableLikeType, VariableLikeType]:
        """
        Obtain the initial structure, coordinates, and lattice parameters from a list of pymatgen structures.

        :param structures: Structures ordered in sequence of run.
        :param progress: Whether to show a progress bar when reading in the structures.

        :returns: A tuple of the initial structure (as
            a :py:class:`pymatgen.core.structure.Structure`), coordinates (as
            a :py:mod:`scipp` array with dimensions of `time`, `atom`, and `dimension`),
            and lattice parameters (as a :py:mod:`scipp` array with dimensions `time`,
            `dimension1`, and `dimension2`).
        """
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

    def get_indices(
        self, structure: 'pymatgen.core.structure.Structure', specie: Union['pymatgen.core.periodic_table.Element',
                                                                            'pymatgen.core.periodic_table.Specie']
    ) -> Tuple[VariableLikeType, VariableLikeType]:
        """
        Determine the framework and mobile indices from a :py:mod:`pymatgen` structure.
        
        :param structure: The initial structure to determine the indices from.
        :param specie: The specie to calculate the diffusivity for.

        :returns: A tuple of the indices for the specie of interest (mobile) and the
            drift (framework) indices.
        """
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


class MDAnalysisParser(Parser):
    """
    Parser for MDAnalysis structures.

    Takes an MDAnalysis.Universe object as an input. 

    :param universe: MDanalysis universe object to be parsed
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note, 
        that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit 
        that is use for the time interval values.
    :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
        output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
        a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
    :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
        ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with 
        a step size the same as the smallest interval.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param distance_unit: The unit of distance used in the input structures. Optional, defaults to angstroms.
    :param sub_sample_atoms: Subsample the atoms in the trajectory. Optional, defaults to 1.
    :param sub_sample_traj: Subsample the trajectory. Optional, defaults to 1.
    :param progress: Whether to show a progress bar when reading in the structures. Optional, defaults to `True`.
    """

    def __init__(self,
                 universe: 'MDAnalysis.core.universe.Universe',
                 specie: str,
                 time_step: VariableLikeType,
                 step_skip: VariableLikeType,
                 dt: VariableLikeType = None,
                 dimension: str = 'xyz',
                 distance_unit: sc.Unit = sc.units.angstrom,
                 specie_indices: VariableLikeType = None,
                 masses: VariableLikeType = None,
                 progress: bool = True):

        self.distance_unit = distance_unit

        structure, coords, latt = self.get_structure_coords_latt(universe, progress)

        if specie is not None:
            indices, drift_indices = self.get_indices(structure, specie)
        elif isinstance(specie_indices, sc.Variable):
            if len(specie_indices.dims) > 1:
                print(
                    'If using the centre of mass calculation functionality please reference the method. doi: 10.1063/5.0260928'
                )
                coords, indices, drift_indices = _get_molecules(structure, coords, specie_indices, masses,
                                                                distance_unit)
            else:
                indices, drift_indices = _get_framework(structure, specie_indices)
        else:
            raise TypeError('Unrecognized type for specie or specie_indices, specie_indices must be a sc.array')

        super().__init__(coords, latt, indices, drift_indices, time_step, step_skip, dt, dimension)

    def get_structure_coords_latt(
        self,
        universe: 'MDAnalysis.core.universe.Universe',
        progress: bool = True,
    ) -> Tuple["MDAnalysis.core.universe.Universe", VariableLikeType, VariableLikeType]:
        """
        Obtain the initial structure, coordinates, and lattice parameters from an MDAnalysis.Universe object.

        :param universe: MDanalysis universe object.
        :param progress: Whether to show a progress bar when reading in the structures.
        :param sub_sample_atoms: Subsample the atoms in the trajectory. Optional, defaults to 1.
        :param sub_sample_traj: Subsample the trajectory. Optional, defaults to 1.

        :returns: A tuple of:  the initial structure (as
            a :py:class:`MDAnalysis.core.universe.Universe`), coordinates (as
            a :py:mod:`scipp` array with dimensions of `time`, `atom`, and `dimension`),
            and lattice parameters (as a :py:mod:`scipp` array with dimensions `time`,
            `dimension1`, and `dimension2`).
        """
        first = True
        coords_l = []
        latt_l = []
        if progress:
            iterator = tqdm(universe.trajectory, desc='Reading Trajectory')
        else:
            iterator = universe.trajectory

        for struct in iterator:
            if first:
                structure = universe.atoms
                first = False
            matrix = np.array(struct.triclinic_dimensions)
            inv_matrix = np.linalg.inv(matrix)
            coords_l.append(np.dot(universe.atoms.positions, inv_matrix))
            latt_l.append(np.array(matrix))

        coords_l = np.array(coords_l)
        latt_l = np.array(latt_l)

        coords = sc.array(dims=['time', 'atom', 'dimension'], values=coords_l, unit=self.distance_unit)
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt_l, unit=self.distance_unit)

        return structure, coords, latt

    def get_indices(
        self,
        structure: "MDAnalysis.universe.Universe",
        specie: str,
    ) -> Tuple[VariableLikeType, VariableLikeType]:
        """
        Determine framework and non-framework indices for an :py:mod:`MDAnalysis` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.

        :return: Tuple containing indices for the atoms in the trajectory used in the calculation of the
            diffusion and indices of framework atoms.
        """
        indices = []
        drift_indices = []

        if not isinstance(specie, list):
            specie = [specie]

        for i, site in enumerate(structure):
            if site.type in specie:
                indices.append(i)
            else:
                drift_indices.append(i)

        if len(indices) == 0:
            raise ValueError("There are no species selected to calculate the mean-squared displacement of.")

        indices = sc.Variable(dims=['atom'], values=indices)
        drift_indices = sc.Variable(dims=['atom'], values=drift_indices)

        return indices, drift_indices


def _get_molecules(structure: Union["ase.atoms.Atoms", "pymatgen.core.structure.Structure",
                                    "MDAnalysis.universe.Universe"], coords: VariableLikeType,
                   indices: VariableLikeType, masses: VariableLikeType,
                   distance_unit: sc.Unit) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Determine framework and non-framework indices for an :py:mod:`ase` or :py:mod:`pymatgen` or :py:mod:`MDAnalysis` compatible file when specie_indices are provided and contain multiple molecules. Warning: This function changes the structure without changing the object.

    :param structure: Initial structure.
    :param coords: fractional coordinates for all atoms
    :param indices: indices for the atoms in the molecules in the trajectory used in the calculation 
    of the diffusion, this must include 2 dimensions 'atom' - The final number of desired atoms and 'atoms_per_group' - the number of atoms in each molecule
    :param masses: Masses associated with indices in indices, dimension should be 'atoms_per_group'.

    :return: Tuple containing: Tuple containing: fractional coordinates for centers and framework atoms
    and Tuple containing: indices for centers used in the calculation 
    of the diffusion and indices of framework atoms.
    """
    drift_indices = []
    n_molecules = len(indices['atoms_per_group', 0])

    if set(indices.dims) != {'atom', 'atoms_per_group'}:
        raise ValueError("Variable must contain only 'atom' and 'atoms_per_group' as dimensions.")

    for i, site in enumerate(structure):
        if i not in indices.values:
            drift_indices.append(i)

    if masses is None:
        weights = sc.ones_like(indices)
    elif len(masses.values) != len(indices['atom', 0]):
        raise ValueError('Masses must be the same length as a molecule or particle group')
    else:
        weights = masses.copy()

    new_s_coords = _calculate_centers_of_mass(coords, weights, indices)

    if coords.dtype == np.float32:
        # MDAnalysis uses float32, so we need to convert to float32 to avoid concat error
        new_s_coords = new_s_coords.astype(np.float32)

    new_coords = sc.concat([new_s_coords, coords['atom', drift_indices]], 'atom')
    new_indices = sc.Variable(dims=['molecule'], values=list(range(n_molecules)))
    new_drift_indices = sc.Variable(dims=['molecule'],
                                    values=list(range(n_molecules, n_molecules + len(drift_indices))))

    return new_coords, new_indices, new_drift_indices


def _calculate_centers_of_mass(
    coords: VariableLikeType,
    weights: VariableLikeType,
    indices: VariableLikeType,
) -> VariableLikeType:
    """
    Calculates the weighted molecular centre of mass based on chosen weights and indices as per  DOI: 10.1063/5.0260928. 
    The method uses the pseudo centre of mass recentering method for efficient centre of mass calculation
    
     :param coords: array of fractional coordinates these should be dimensionless
     :param weights: 1D array of weights of elements within molecule
     :param indices: Scipp array of indices for the atoms in the molecules in the trajectory used in the calculation 
    of the diffusion, this must include 2 dimensions 'atom' - The final number of desired atoms and 'atoms_per_group' - the number of atoms in each molecule

     :return: Array containing coordinates of centres of mass of molecules
    """
    s_coords = sc.fold(coords['atom', indices.values.flatten()], 'atom', dims=indices.dims, shape=indices.shape)
    theta = s_coords * (2 * np.pi * (sc.units.rad))
    xi = sc.cos(theta)
    zeta = sc.sin(theta)
    dims_id = 'atoms_per_group'
    xi_bar = (weights * xi).sum(dim=dims_id) / weights.sum(dim=dims_id)
    zeta_bar = (weights * zeta).sum(dim=dims_id) / weights.sum(dim=dims_id)
    theta_bar = sc.atan2(y=-zeta_bar, x=-xi_bar) + np.pi * sc.units.rad
    new_s_coords = theta_bar / (2 * np.pi * (sc.units.rad))

    #Implementation of pseudo-centre of mass approach to centre of mass calculation (see DOI:10.1063/5.0260928 ).
    pseudo_com_recentering = ((s_coords - (new_s_coords + 0.5)) % 1)
    com_pseudo_space = (weights * pseudo_com_recentering).sum(dim=dims_id) / weights.sum(dim=dims_id)
    corrected_com = ((com_pseudo_space + (new_s_coords + 0.5)) % 1)

    return corrected_com


def _get_framework(structure: Union["ase.atoms.Atoms", "pymatgen.core.structure.Structure",
                                    "MDAnalysis.universe.Universe"],
                   indices: VariableLikeType) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine the framework indices from an :py:mod:`ase` or :py:mod:`pymatgen` or :py:mod:`MDAnalysis` compatible file when indices are provided
    
    :param structure: Initial structure.
    :param indices: Indices for the atoms in the trajectory used in the calculation of the 
        diffusion.
    :param framework_indices: Indices of framework to be used in drift correction. If set to None will return all indices that are not in indices.
    
    :return: Tuple containing: indices for the atoms in the trajectory used in the calculation of the
        diffusion and indices of framework atoms. 
    """

    drift_indices = []

    for i, site in enumerate(structure):
        if i not in indices:
            drift_indices.append(i)

    drift_indices = sc.Variable(dims=['atom'], values=drift_indices)

    return indices, drift_indices
