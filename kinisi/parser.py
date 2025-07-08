"""
Parsers for kinisi. This module is responsible for reading in input files from :py:mod:`pymatgen`,
:py:mod:`MDAnalysis`, and :py:mod:`ase`.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61) and Harry Richardson (Harry-Rich).

from typing import Tuple, Union
import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType

DIMENSIONALITY = {
    "x": np.s_[0],
    "y": np.s_[1],
    "z": np.s_[2],
    "xy": np.s_[:2],
    "xz": np.s_[::2],
    "yz": np.s_[1:],
    "xyz": np.s_[:],
    b"x": np.s_[0],
    b"y": np.s_[1],
    b"z": np.s_[2],
    b"xy": np.s_[:2],
    b"xz": np.s_[::2],
    b"yz": np.s_[1:],
    b"xyz": np.s_[:],
}


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
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param progress: Whether to show a progress bar when reading in the structures. Optional, defaults to `True`.
    """

    def __init__(
        self,
        snapshots: Union[
            "pymatgen.core.structure.Structure", "MDAnalysis.core.universe.Universe"
        ],
        specie: Union[
            "pymatgen.core.periodic_table.Element",
            "pymatgen.core.periodic_table.Specie",
            "str",
        ],
        time_step: VariableLikeType,
        step_skip: VariableLikeType,
        dt: VariableLikeType = None,
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: VariableLikeType = None,
        masses: VariableLikeType = None,
        dimension: str = "xyz",
        progress: bool = True,
    ):
        self.time_step = time_step
        self.step_skip = step_skip
        self._dimension = dimension
        self.dt = dt
        self.distance_unit = distance_unit

        structure, coords, latt = self.get_structure_coords_latt(snapshots, progress)

        self.create_integer_dt(coords, time_step, step_skip)

        indices, drift_indices = self.generate_indices(
            structure, specie_indices, coords, specie, masses
        )

        structure, coords, latt = self.get_structure_coords_latt(snapshots, progress)

        self.create_integer_dt(coords, time_step, step_skip)

        indices, drift_indices = self.generate_indices(structure, specie_indices, coords, specie, masses)

        self.indices = indices
        self.drift_indices = drift_indices

        disp = self.calculate_displacements(coords, latt)
        drift_corrected = self.correct_drift(disp)

        self._slice = DIMENSIONALITY[dimension.lower()]
        drift_corrected = drift_corrected["dimension", self._slice]
        self.dimensionality = (
            drift_corrected.sizes["dimension"] * sc.units.dimensionless
        )

        self.displacements = drift_corrected["atom", indices]
        self._volume = np.prod(latt.values[0].diagonal()) * self.distance_unit**3

    def create_integer_dt(
        self,
        coords: VariableLikeType,
        time_step: VariableLikeType,
        step_skip: VariableLikeType,
    ) -> None:
        """
        Create an integer time interval from the given time intervals (and if necessary the time interval object).

        :param coords: The fractional coordiates of the atoms in the trajectory. This should be a :py:mod:`scipp`
            array type object with dimensions of 'atom', 'time', and 'dimension'.
        :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note,
            that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit
            that is use for the time interval values.
        :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
            output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
            a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
        """
        if self.dt is None:
            self.dt_index = sc.arange(
                start=1, stop=coords.sizes["time"], step=1, dim="time interval"
            )
            self.dt = self.dt_index * time_step * step_skip
        self.dt_index = (self.dt / (time_step * step_skip)).astype(int)

    def generate_indices(
        self,
        structure: Tuple[
            Union[
                "pymatgen.core.structure.Structure", "MDAnalysis.core.universe.Universe"
            ],
            VariableLikeType,
            VariableLikeType,
        ],
        specie_indices: VariableLikeType,
        coords: VariableLikeType,
        specie: Union[
            "pymatgen.core.periodic_table.Element",
            "pymatgen.core.periodic_table.Specie",
            "str",
        ],
        masses: VariableLikeType,
    ) -> Tuple[VariableLikeType, VariableLikeType]:
        """
        Handle the specie indices and determine the indices for the framework and drift correction.

        :param structure: The initial structure to determine the indices from.
        :param specie_indices: Indices for the atoms in the trajectory used in the diffusion calculation
        :param coords: The fractional coordinates of the atoms in the trajectory.
        :param specie: The specie to calculate the diffusivity for.
        :param masses: Masses associated with indices in indices.

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
            raise TypeError(
                "Unrecognized type for specie or specie_indices, specie_indices must be a sc.array"
            )
        return indices, drift_indices

    def calculate_displacements(
        self, coords: VariableLikeType, lattice: VariableLikeType
    ) -> VariableLikeType:
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
            values=np.einsum("jik,jkl->jil", coords.values, lattice.values),
            unit=coords.unit,
        )
        wrapped_diff = sc.array(
            dims=["obs"] + list(coords.dims[1:]),
            values=(wrapped["time", 1:] - wrapped["time", :-1]).values,
            unit=coords.unit,
        )
        diff_diff = sc.array(
            dims=wrapped_diff.dims,
            values=np.einsum(
                "jik,jkl->jil",
                np.floor(
                    np.einsum("jik,jkl->jil", wrapped_diff.values, lattice_inv[1:])
                    + 0.5
                ),
                lattice.values[1:],
            ),
            unit=coords.unit,
        )
        unwrapped_diff = wrapped_diff - diff_diff
        return sc.cumsum(unwrapped_diff, "obs")

    def correct_drift(self, disp: VariableLikeType) -> VariableLikeType:
        """
        Perform drift correction, such that the displacement is calculated normalised to any framework drift.

        :param disp: Displacements for all atoms in the simulation. A :py:mod:`scipp` array with dimensions
            of `obs`, `atom` and `dimension`.

        :return: Displacements corrected to account for drift of a framework.
        """
        if self.drift_indices.size > 0:
            return disp - sc.mean(disp["atom", self.drift_indices.values], "atom")
        else:
            return disp


def get_molecules(
    structure: Union[
        "ase.atoms.Atoms",
        "pymatgen.core.structure.Structure",
        "MDAnalysis.universe.Universe",
    ],
    coords: VariableLikeType,
    indices: VariableLikeType,
    masses: VariableLikeType,
    distance_unit: sc.Unit,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    try:
        indices = indices - 1
    except:
        raise ValueError("Molecules must be of same length")

    n_molecules = indices.shape[0]

    # Removed method for framework_indices
    for i, site in enumerate(structure):
        if i not in indices.values:
            drift_indices.append(i)

    if masses is None:
        weights = sc.ones_like(indices)
    elif len(masses.values) != indices.values.shape[1]:
        raise ValueError("Masses must be the same length as a molecule")
    else:
        weights = masses.copy()

    new_s_coords = _calculate_centers_of_mass(coords, weights, indices, distance_unit)

    if coords.dtype == np.float32:
        # MDAnalysis uses float32, so we need to convert to float32 to avoid concat error
        new_s_coords = new_s_coords.astype(np.float32)

    new_coords = sc.concat([new_s_coords, coords["atom", drift_indices]], "atom")
    new_indices = sc.Variable(dims=["molecule"], values=list(range(n_molecules)))
    new_drift_indices = sc.Variable(
        dims=["molecule"],
        values=list(range(n_molecules, n_molecules + len(drift_indices))),
    )

    return new_coords, new_indices, new_drift_indices


def get_framework(
    structure: Union[
        "ase.atoms.Atoms",
        "pymatgen.core.structure.Structure",
        "MDAnalysis.universe.Universe",
    ],
    indices: VariableLikeType,
) -> Tuple[np.ndarray, np.ndarray]:
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

    drift_indices = sc.Variable(dims=["atom"], values=drift_indices)

    return indices, drift_indices


def _calculate_centers_of_mass(
    coords: VariableLikeType,
    weights: VariableLikeType,
    indices: VariableLikeType,
    distance_unit: sc.Unit,
) -> VariableLikeType:
    """
    Calculates the weighted molecular centre of mass based on chosen weights and indices as per https://doi.org/10.1080/2151237X.2008.10129266
    The method involves projection of the each coordinate onto a circle to allow for efficient COM calculation

     :param coords: array of coordinates
     :param weights: 1D array of weights of elements within molecule
     :param indices: N by M dimensional array of indices of N molecules of M atoms

     :return: Array containing coordinates of centres of mass of molecules
    """
    s_coords = sc.fold(
        coords["atom", indices.values.flatten()],
        "atom",
        dims=indices.dims,
        shape=indices.shape,
    )
    theta = s_coords * (2 * np.pi * (sc.units.rad / distance_unit))
    xi = sc.cos(theta)
    zeta = sc.sin(theta)
    # This allows the dimensions of the indices to be any word, paired with 'atom'.
    dims_id = [i for i in indices.dims if i != "atom"][0]
    xi_bar = (weights * xi).sum(dim=dims_id) / weights.sum(dim=dims_id)
    zeta_bar = (weights * zeta).sum(dim=dims_id) / weights.sum(dim=dims_id)
    theta_bar = sc.atan2(y=-zeta_bar, x=-xi_bar) + np.pi * sc.units.rad
    new_s_coords = theta_bar / (2 * np.pi * (sc.units.rad / distance_unit))
    return new_s_coords
