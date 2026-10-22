"""
The :py:class:`ASEParser` class is a parser for ASE atoms object.
It is used to extract the necessary data for diffusion analysis from ASE.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Josh Dunn (jd15489).

import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType
from tqdm import tqdm

from kinisi import __version__
from kinisi.parser import Parser

from .due import Doi, due


class ASEParser(Parser):
    """
    Parser for a trajectory of :py:class:`ase.atoms.Atoms` objects, such as an :py:class:`ase.io.trajectory.Trajectory`.

    :param atoms: Atoms objects ordered in sequence of run.
    :param specie: Specie to calculate diffusivity for as a String, list of strings, or :py:mod:`scipp` array of
        strings, e.g. :py:attr:`'Li'`. Either this or :py:attr:`specie_indices` must be given, not both.
    :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator.
        This must be given as a :py:mod:`scipp`-type scalar, and its unit is used for the time interval values.
    :param step_skip: Sampling frequency of the simulation trajectory, i.e., how many time steps exist between the
        output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
        a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
    :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
        ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with
        a step size the same as the smallest interval.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param distance_unit: The unit of distance used in the input structures. Optional, defaults to angstroms.
    :param specie_indices: Indices of the atoms to calculate the diffusivity for, as a :py:mod:`scipp` array.
        A 1D array with dimension `particle` selects individual atoms. A 2D array with dimensions `particle` and
        `atoms in particle` groups atoms into molecules, and the centre of mass of each molecule is used.
        Optional, defaults to `None`, in which case :py:attr:`specie` is used.
    :param drift_indices: Indices of the atoms used for drift correction, as a :py:mod:`scipp` array. Optional,
        defaults to all atoms that are not being analysed.
    :param masses: Masses used to weight the centre of mass of each molecule, as a :py:mod:`scipp` array with
        dimension `atoms in particle`. Optional, defaults to `None`, in which case all atoms are weighted equally.
    :param progress: Whether to show a progress bar when reading in the structures. Optional, defaults to `True`.
    """

    @due.dcite(
        Doi('10.1088/1361-648X/aa680e'),
        path='kinisi.ase.ASEParser',
        description='Atomic simulation environment which has been used for parsing.',
        version=__version__,
    )
    def __init__(
        self,
        atoms: list['ase.atoms.Atoms'],
        specie: str | list | VariableLikeType,
        time_step: VariableLikeType,
        step_skip: VariableLikeType,
        dt: VariableLikeType = None,
        dimension: str = 'xyz',
        distance_unit: sc.Unit = sc.units.angstrom,
        specie_indices: VariableLikeType = None,
        drift_indices: VariableLikeType = None,
        masses: VariableLikeType = None,
        progress: bool = True,
    ):
        atoms, coords, latt = self.get_structure_coords_latt(atoms, distance_unit, progress)

        specie_indices, drift_indices = super().get_specie_and_drift_indices(
            specie, specie_indices, drift_indices, atoms
        )

        super().__init__(
            coords=coords,
            latt=latt,
            time_step=time_step,
            step_skip=step_skip,
            dt=dt,
            specie_indices=specie_indices,
            drift_indices=drift_indices,
            masses=masses,
            dimension=dimension,
        )

    def get_structure_coords_latt(
        self, atoms: list['ase.atoms.Atoms'], distance_unit: sc.Unit, progress: bool = True
    ) -> tuple['ase.atoms.Atoms', VariableLikeType, VariableLikeType]:
        """
        Obtain the initial structure, fractional coordinates, and lattice matrices from a list of
        :py:class:`ase.atoms.Atoms` objects.

        :param atoms: Atoms objects ordered in sequence of run.
        :param distance_unit: The unit of distance used in the input structures.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.

        :return: Tuple containing: initial structure, fractional coordinates for all atoms, and lattice descriptions.
        """
        coords, latt = [], []
        first = True
        if progress:
            iterator = tqdm(atoms, desc='Reading Trajectory')
        else:
            iterator = atoms
        for struct in iterator:
            if first:
                structure = struct
                first = False
            scaled_positions = struct.get_scaled_positions()
            coords.append(np.array(scaled_positions))
            latt.append(struct.cell[:])

        coords = sc.array(dims=['time', 'particle', 'dimension'], values=coords, unit=sc.units.dimensionless)
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt, unit=distance_unit)

        return structure, coords, latt

    def get_indices(
        self,
        structure: 'ase.atoms.Atoms',
        specie: str | list | VariableLikeType,
    ) -> tuple[VariableLikeType, VariableLikeType]:
        """
        Determine framework and non-framework indices for a :py:mod:`ase` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, list of strings, or scipp.Variable of strings.

        :returns: Tuple containing: indices for the atoms in the trajectory used in the calculation of the diffusion
            and indices of framework atoms.
        """
        indices = []
        drift_indices = []

        if isinstance(specie, str):
            specie = [specie]
        if isinstance(specie, list):
            specie = sc.array(dims=['specie'], values=specie)

        for i, site in enumerate(structure):
            if site.symbol in specie.values:
                indices.append(i)
            else:
                drift_indices.append(i)

        if len(indices) == 0:
            raise ValueError('There are no species selected to calculate the mean-squared displacement of.')

        indices = sc.Variable(dims=['particle'], values=indices)
        drift_indices = sc.Variable(dims=['particle'], values=drift_indices)

        return indices, drift_indices

    def get_drift_indices(
        self,
        structure: 'ase.atoms.Atoms',
        specie_indices: VariableLikeType,
    ) -> VariableLikeType:
        """
        Determine framework indices for an :py:mod:`ase` structure.

        :param structure: Initial structure.

        :return: Indices for the atoms in the trajectory used as framework atoms.
        """
        return sc.array(dims=['particle'], values=[x for x in range(len(structure)) if x not in specie_indices.values])
