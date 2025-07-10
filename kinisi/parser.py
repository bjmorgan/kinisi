"""
Parsers for kinisi. This module is responsible for reading in input files from :py:mod:`pymatgen`,
:py:mod:`MDAnalysis`, and :py:mod:`ase`.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61) and Harry Richardson (Harry-Rich).

from typing import Union

import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType

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
    b'xyz': np.s_[:],
}

EINSUM_DIMENSIONS = {
    'time': 't',
    'atom': 'a',
    'image': 'i',
    'row': 'r',
    'column': 'c'
} # Single letter labels to be used as subscripts for dimensions of scipp arrays in einsums.


class Parser:
    """
    The base class for object parsing.

    :param snapshots: The snapshots from the trajectory given the positions of atoms.
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
    :param distance_unit: The unit of distance used in the input structures. Optional, defaults to angstroms.
    :param specie_indices: Indices of the specie to calculate the diffusivity for. Optional, defaults to `None`.
    :param masses: Masses of the atoms in the structure. Optional, defaults to `None`.
        If used should be a 1D scipp array of dimension 'group_of_atoms'.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param progress: Whether to show a progress bar when reading in the structures. Optional, defaults to `True`.
    """

    def __init__(
        self,
        snapshots: Union['pymatgen.core.structure.Structure', 'MDAnalysis.core.universe.Universe'],
        specie: Union[
            'pymatgen.core.periodic_table.Element',
            'pymatgen.core.periodic_table.Specie',
            'str',
        ],
        time_step: VariableLikeType,
        step_skip: VariableLikeType,
        dt: VariableLikeType = None,
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: VariableLikeType = None,
        masses: VariableLikeType = None,
        dimension: str = 'xyz',
        progress: bool = True,
        old_calc_disps: bool = False,
    ):
        self.time_step = time_step
        self.step_skip = step_skip
        self._dimension = dimension
        self.dt = dt
        self.distance_unit = distance_unit

        structure, coords, latt = self.get_structure_coords_latt(snapshots, progress)

        self.dt_index = self.create_integer_dt(coords, time_step, step_skip)

        coords, indices, drift_indices = self.generate_indices(structure, specie_indices, coords, specie, masses)

        self.indices = indices
        self.drift_indices = drift_indices
        self._coords = coords

        if old_calc_disps:
            disp = self.old_calculate_displacements(coords, latt)
        else:
            disp = self.calculate_displacements(coords, latt)
        self._disp = disp
        drift_corrected = self.correct_drift(disp)

        self._slice = DIMENSIONALITY[dimension.lower()]
        drift_corrected = drift_corrected['dimension', self._slice]
        self.dimensionality = drift_corrected.sizes['dimension'] * sc.units.dimensionless

        self.displacements = drift_corrected['atom', indices]
        self._volume = np.prod(latt.values[0].diagonal()) * self.distance_unit**3

    def create_integer_dt(
        self,
        coords: VariableLikeType,
        time_step: VariableLikeType,
        step_skip: VariableLikeType,
    ) -> VariableLikeType:
        """
        Create an integer time interval from the given time intervals (and if necessary the time interval object).
        Also checks that the time intervals provided in the dt parameter are a valid subset of the simulation time
        intervals.

        :param coords: The fractional coordiates of the atoms in the trajectory. This should be a :py:mod:`scipp`
            array type object with dimensions of 'atom', 'time', and 'dimension'.
        :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note,
            that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit
            that is use for the time interval values.
        :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
            output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
            a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.

        :raises ValueError: If the time intervals provided in the dt parameter are not a subset of the time intervals
            present in the simulation, based on the time_step and step_skip parameters and number of snapshots
            in the trajectory.

        :return: The integer time intervals as a :py:mod:`scipp` array with dimensions of 'time interval'.
        """
        dt_all = sc.arange(start=1, stop=coords.sizes['time'], step=1, dim='time interval') * time_step * step_skip
        if self.dt is not None:
            if not is_subset_approx(self.dt.values, dt_all.values):
                raise ValueError(
                    'The time intervals provided in the dt parameter are not a subset of the time intervals '
                    'present in the simulation, based on the time_step and step_skip parameters and number of '
                    'snapshots in the trajectory.'
                )
        else:
            dt_index = sc.arange(start=1, stop=coords.sizes['time'], step=1, dim='time interval')
            self.dt = dt_index * time_step * step_skip

        dt_index = (self.dt / (time_step * step_skip)).astype(int)
        return dt_index

    def generate_indices(
        self,
        structure: tuple[
            Union['pymatgen.core.structure.Structure', 'MDAnalysis.core.universe.Universe'],
            VariableLikeType,
            VariableLikeType,
        ],
        specie_indices: VariableLikeType,
        coords: VariableLikeType,
        specie: Union[
            'pymatgen.core.periodic_table.Element',
            'pymatgen.core.periodic_table.Specie',
            'str',
        ],
        masses: VariableLikeType,
    ) -> tuple[VariableLikeType, VariableLikeType]:
        """
        Handle the specie indices and determine the indices for the framework and drift correction.

        :param structure: The initial structure to determine the indices from.
        :param specie_indices: Indices for the atoms in the trajectory used in the diffusion calculation
        :param coords: The fractional coordinates of the atoms in the trajectory.
        :param specie: The specie to calculate the diffusivity for.
        :param masses: Masses associated with indices in indices. 1D scipp array of dim 'group_of_atoms'

        :return: A tuple containing the indices for the atoms in the trajectory used in the diffusion calculation
            and indices of framework atoms.
        """
        if specie is not None:
            indices, drift_indices = self.get_indices(structure, specie)
        elif isinstance(specie_indices, sc.Variable):
            if len(specie_indices.dims) > 1:
                coords, indices, drift_indices = get_molecules(
                    structure, coords, specie_indices, masses, self.distance_unit
                )
            else:
                indices, drift_indices = get_framework(structure, specie_indices)
        else:
            raise TypeError('Unrecognized type for specie or specie_indices, specie_indices must be a sc.array')
        return coords, indices, drift_indices

    def old_calculate_displacements(self, coords: VariableLikeType, lattice: VariableLikeType) -> VariableLikeType:
        """
        Calculate the absolute displacements of the atoms in the trajectory.

        :param coords: The fractional coordiates of the atoms in the trajectory. This should be a :py:mod:`scipp`
            array type object with dimensions of 'atom', 'time', and 'dimension'.
        :param lattice: A series of matrices that describe the lattice in each step in the trajectory.
            A :py:mod:`scipp` array with dimensions of 'time', 'dimension1', and 'dimension2'.

        :return: The absolute displacements of the atoms in the trajectory.
        """
        lattice_inv = np.linalg.inv(lattice.values)
        wrapped = sc.array(
            dims=coords.dims,
            values=np.einsum('jik,jkl->jil', coords.values, lattice.values),
            unit=lattice.unit,
        )
        wrapped_diff = sc.array(
            dims=['obs'] + list(coords.dims[1:]),
            values=(wrapped['time', 1:] - wrapped['time', :-1]).values,
            unit=lattice.unit,
        )
        diff_diff = sc.array(
            dims=wrapped_diff.dims,
            values=np.einsum(
                'jik,jkl->jil',
                np.floor(np.einsum('jik,jkl->jil', wrapped_diff.values, lattice_inv[1:]) + 0.5),
                lattice.values[1:],
            ),
            unit=lattice.unit,
        )
        unwrapped_diff = wrapped_diff - diff_diff
        return sc.cumsum(unwrapped_diff, 'obs')
    
    def calculate_displacements(self, coords: VariableLikeType, lattice: VariableLikeType) -> VariableLikeType:
        """
        Calculate the absolute displacements of the atoms in the trajectory. This is done by finding the minimum cartesian 
            displacement vector, from its 8 periodic images. This ensures that triclinic cells are treated correctly.
        
        :param coords: The fractional coordiates of the atoms in the trajectory. This should be a :py:mod:`scipp`
            array type object with dimensions of 'atom', 'time', and 'dimension'.
        :param lattice: A series of matrices that describe the lattice in each step in the trajectory.
            A :py:mod:`scipp` array with dimensions of 'time', 'row', and 'column'.
            
        :return: The absolute displacements of the atoms in the trajectory.
        """
        diff = np.diff(coords.values, axis=0)
        images = np.tile([[0,0,0],[-1,0,0],[-1,-1,0],[0,-1,0],[0,0,1],[-1,0,1],[-1,-1,1],[0,-1,1]], (diff.shape[0],diff.shape[1],1,1))

        diff[diff < 0] += 1
        images = images + diff[..., np.newaxis, :]

        cart_images = np.einsum('taid,tdc->taid', images, lattice.values[1:])
        image_disps = np.linalg.norm(cart_images, axis=-1)
        min_index = np.argmin(image_disps, axis=-1)

        min_vectors = cart_images[np.arange(images.shape[0])[:, None], np.arange(images.shape[1])[None, :], min_index]
        min_vectors = sc.array(dims=['obs'] + list(coords.dims[1:]),
                               values=min_vectors,
                               unit=coords.unit)
        disps = sc.cumsum(min_vectors, 'obs')

        return disps

    def correct_drift(self, disp: VariableLikeType) -> VariableLikeType:
        """
        Perform drift correction, such that the displacement is calculated normalised to any framework drift.

        :param disp: Displacements for all atoms in the simulation. A :py:mod:`scipp` array with dimensions
            of `obs`, `atom` and `dimension`.

        :return: Displacements corrected to account for drift of a framework.
        """
        if self.drift_indices.size > 0:
            return disp - sc.mean(disp['atom', self.drift_indices.values], 'atom')
        else:
            return disp

    @property
    def coords(self):
        '''
        Coordinates of 'atoms', this may be the raw coordinates parsed or centres of mass/geometry.
        '''
        return self._coords
    
    @property
    def disp(self):
        '''
        Atom displacements, without drift correction.
        '''
        return self._disp

def get_molecules(
    structure: Union[
        'ase.atoms.Atoms',
        'pymatgen.core.structure.Structure',
        'MDAnalysis.universe.Universe',
    ],
    coords: VariableLikeType,
    indices: VariableLikeType,
    masses: VariableLikeType,
    distance_unit: sc.Unit,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Determine framework and non-framework indices for an :py:mod:`ase` or :py:mod:`pymatgen` or :py:mod:`MDAnalysis` compatible file when
    specie_indices are provided and contain multiple molecules. Warning: This function changes the structure without changing the object.

    :param structure: Initial structure.
    :param coords: fractional coordinates for all atoms.
    :param indices: indices for the atoms in the molecules in the trajectory used in the calculation
    of the diffusion.
    :param masses: Masses associated with indices in indices.
    :param framework_indices: Indices of framework to be used in drift correction. If set to None will return all indices that are not in indices.


    :return: Tuple containing: Tuple containing: fractional coordinates for centers and framework atoms
    and Tuple containing: indices for centers used in the calculation
    of the diffusion and indices of framework atoms.
    """
    drift_indices = []

    if set(indices.dims) != {'atom', 'group_of_atoms'}:
        raise ValueError("indices must contain only 'atom' and 'group_of_atoms' as dimensions.")
    
    n_molecules = indices.sizes['group_of_atoms']

    for i, site in enumerate(structure):
        if i not in indices.values:
            drift_indices.append(i)

    if masses is None:
        weights = sc.ones_like(indices)
    elif len(masses.values) != len(indices['atom', 0]):
        raise ValueError('Masses must be the same length as a molecule or particle group')
    else:
        weights = masses.copy()

    if 'group_of_atoms' not in weights.dims:
        raise ValueError("masses must contain 'group_of_atoms' as dimensions.")

    new_s_coords = _calculate_centers_of_mass(coords, weights, indices)

    if coords.dtype == np.float32:
        # MDAnalysis uses float32, so we need to convert to float32 to avoid concat error
        new_s_coords = new_s_coords.astype(np.float32)

    new_coords = sc.concat([new_s_coords, coords['atom', drift_indices]], 'atom')
    new_indices = sc.Variable(dims=['atom'], values=list(range(n_molecules)))
    new_drift_indices = sc.Variable(
        dims=['atom'],
        values=list(range(n_molecules, n_molecules + len(drift_indices))),
    )

    return new_coords, new_indices, new_drift_indices


def get_framework(
    structure: Union[
        'ase.atoms.Atoms',
        'pymatgen.core.structure.Structure',
        'MDAnalysis.universe.Universe',
    ],
    indices: VariableLikeType,
) -> tuple[np.ndarray, np.ndarray]:
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

    for i, _site in enumerate(structure):
        if i not in indices:
            drift_indices.append(i)

    drift_indices = sc.Variable(dims=['atom'], values=drift_indices)

    return indices, drift_indices


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
     :param indices: Scipp array of indices for the atoms in the molecules in the trajectory,
     this must include 2 dimensions 'atom' - The final number of desired atoms and 'group_of_atoms' - the number of atoms in each molecule

     :return: Array containing coordinates of centres of mass of molecules
    """
    s_coords = sc.fold(coords['atom', indices.values.flatten()], 'atom', dims=indices.dims, shape=indices.shape)
    theta = s_coords * (2 * np.pi * (sc.units.rad))
    xi = sc.cos(theta)
    zeta = sc.sin(theta)
    dims_id = 'group_of_atoms'
    xi_bar = (weights * xi).sum(dim=dims_id) / weights.sum(dim=dims_id)
    zeta_bar = (weights * zeta).sum(dim=dims_id) / weights.sum(dim=dims_id)
    theta_bar = sc.atan2(y=-zeta_bar, x=-xi_bar) + np.pi * sc.units.rad
    new_s_coords = theta_bar / (2 * np.pi * (sc.units.rad))

    pseudo_com_recentering = (s_coords - (new_s_coords + 0.5)) % 1
    com_pseudo_space = (weights * pseudo_com_recentering).sum(dim=dims_id) / weights.sum(dim=dims_id)
    corrected_com = (com_pseudo_space + (new_s_coords + 0.5)) % 1

    print('If using the kinisi centre of mass feature, please reference: DOI: 10.1063/5.0260928')
    return corrected_com


def is_subset_approx(B: np.array, A: np.array, tol: float = 1e-9) -> bool:
    """
    Check if all elements in B are approximately equal to any element in A within a tolerance.
    This is useful for comparing floating-point numbers where exact equality is not feasible.

    :param B: The array to check if it is a subset of A.
    :param A: The array to check against.
    :param tol: The tolerance for comparison. Default is 1e-9.

    :return: True if all elements in B are approximately equal to any element in A, False otherwise.
    """
    return all(any(abs(a - b) < tol for a in A) for b in B)
