"""
The `PymatgenParser` class is a parser for pymatgen structures for :py:mod:`kinisi`.
It is used to extract the necessary data for diffusion analysis from a list of pymatgen structures.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61), Harry Richardson (Harry-Rich) and Josh Dunn (jd15489).

import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType
from tqdm import tqdm

from kinisi import __version__
from kinisi.parser import Parser

from .due import Doi, due


class PymatgenParser(Parser):
    """
    Parser for a list of :py:class:`pymatgen.core.structure.Structure` objects.

    :param structures: Structures ordered in sequence of run.
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
        Doi('10.1016/j.commatsci.2012.10.028'),
        path='kinisi.pymatgen.PymatgenParser',
        description='Pymatgen, which as been used in parsing.',
        version=__version__,
    )
    def __init__(
        self,
        structures: list['pymatgen.core.structure.Structure'],
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
        structure, coords, latt = self.get_structure_coords_latt(structures, distance_unit, progress)

        specie_indices, drift_indices = super().get_specie_and_drift_indices(
            specie, specie_indices, drift_indices, structure
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
        self,
        structures: list['pymatgen.core.structure.Structure'],
        distance_unit: VariableLikeType,
        progress: bool = True,
    ) -> tuple['pymatgen.core.structure.Structure', VariableLikeType, VariableLikeType]:
        """
        Obtain the initial structure, coordinates, and lattice parameters from a list of pymatgen structures.

        :param structures: Structures ordered in sequence of run.
        :param distance_unit: The unit of distance used in the input structures.
        :param progress: Whether to show a progress bar when reading in the structures.

        :returns: A tuple of the initial structure (as
            a :py:class:`pymatgen.core.structure.Structure`), coordinates (as
            a :py:mod:`scipp` array with dimensions of `time`, `particle`, and `dimension`),
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

        coords = sc.array(dims=['time', 'particle', 'dimension'], values=coords_l, unit=sc.units.dimensionless)
        latt = sc.array(dims=['time', 'dimension1', 'dimension2'], values=latt_l, unit=distance_unit)

        return structure, coords, latt

    def get_indices(
        self,
        structure: 'pymatgen.core.structure.Structure',
        specie: str | list | VariableLikeType,
    ) -> tuple[VariableLikeType, VariableLikeType]:
        """
        Determine the framework and mobile indices from a :py:mod:`pymatgen` structure.

        :param structure: The initial structure to determine the indices from.
        :param specie: The specie to calculate the diffusivity for as a String, list of strings, or scipp.Variable of strings.

        :returns: A tuple of the indices for the specie of interest (mobile) and the
            drift (framework) indices.
        """
        indices = []
        drift_indices = []

        if isinstance(specie, str):
            specie = [specie]
        if isinstance(specie, list):
            specie = sc.array(dims=['specie'], values=specie)

        for i, site in enumerate(structure):
            if site.specie.__str__() in specie.values:
                indices.append(i)
            else:
                drift_indices.append(i)
        indices = sc.Variable(dims=['particle'], values=indices)
        drift_indices = sc.Variable(dims=['particle'], values=drift_indices)
        return indices, drift_indices

    def get_drift_indices(
        self,
        structure: 'pymatgen.core.structure.Structure',
        specie_indices: VariableLikeType,
    ) -> VariableLikeType:
        """
        Determine framework indices for an :py:mod:`pymatgen` structure.

        :param structure: Initial structure.

        :return: Indices for the atoms in the trajectory used as framework atoms.
        """
        return sc.array(dims=['particle'], values=[x for x in range(len(structure)) if x not in specie_indices.values])
