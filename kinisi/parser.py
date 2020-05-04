"""
Parser functions, including implementation for :py:mod:`pymatgen` compatible VASP files and :py:mod:`MDAnalysis` compatible trajectories.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0902,R0913

# This parser borrows heavily from the
# pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer
# class, originally authored by Will Richards
# (wrichard@mit.edu) and Shyue Ping Ong.
# We include this statement to not that we make
# no claim to authorship of that code and make
# no attack on the original authors.
#
# In fact, we love pymatgen!

import numpy as np
from pymatgen.analysis.diffusion_analyzer import get_conversion_factor
from pymatgen.core.lattice import Lattice
from pymatgen.core import Structure
import periodictable as pt
from tqdm import tqdm

class PymatgenParser:
    """
    A parser for pymatgen Xdatcar files.
    
    Attributes:
        time_step (:py:attr:`float`): Time step between measurements.
        step_step (:py:attr:`int`): Sampling freqency of the displacements (time_step is multiplied by this number to get the real time between measurements).
        indices (:py:attr:`array_like`): Indices for the atoms in the trajectory used in the calculation of the diffusion.
        delta_t (:py:attr:`array_like`):  Timestep values.
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): Each element in the :py:attr:`list` has the axes [atom, displacement observation, dimension] and there is one element for each delta_t value. *Note: it is necessary to use a :py:attr:`list` of :py:attr:`array_like` as the number of observations is not necessary the same at each timestep point*.
        disp_store (:py:attr:`list` of :py:attr:`array_like`): The :py:attr:`disp_3d` object, with the dimension axes removed by summation through this.
    
    Args:
        structures (:py:attr:`list` or :py:class`pymatgen.core.structure.Structure`): Structures ordered in sequence of run. 
        specie (:py:class:`pymatgen.core.periodic_table.Element` or :py:class:`pymatgen.core.periodic_table.Specie`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        time_step (:py:attr:`float`): Time step between measurements.
        step_step (:py:attr:`int`): Sampling freqency of the displacements (time_step is multiplied by this number to get the real time between measurements).
        min_obs (:py:attr:`int`, optional): Minimum number of observations to have before including in the MSD vs dt calculation. E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times. Defaults to :py:attr:`30`.    
    """
    def __init__(self, structures, specie, time_step, step_skip, min_obs=30):
        self.time_step = time_step
        self.step_skip = step_skip
        self.indices = []

        structure, disp = _get_structure_and_disp(structures)

        drift_corrected = self.correct_for_drift(structure, disp, specie)

        nsteps = drift_corrected.shape[1]

        timesteps = self.smoothed_timesteps(nsteps, min_obs)

        self.delta_t, self.disp_store, self.disp_3d = self.get_disps(
            timesteps, drift_corrected)

    def correct_for_drift(self, structure, disp, specie):
        """
        Perform drift correction

        Args:
            structure (:py:class:`pymatgen.core.structure.Structure`): Initial structure.
            disp (:py:attr:`array_like`): Numpy array of with shape [site, time step, axis].
            specie (:py:class:`pymatgen.core.periodic_table.Element` or :py:class:`pymatgen.core.periodic_table.Specie`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.

        Returns:
            :py:attr:`array_like`: Drift of framework corrected disp.
        """
        framework_indices = []
        for i, site in enumerate(structure):
            if site.specie.symbol == specie:
                self.indices.append(i)
            else:
                framework_indices.append(i)

        # drift corrected position
        if len(framework_indices) > 0:
            framework_disp = disp[framework_indices]
            drift_corrected = disp - np.average(framework_disp, axis=0)[None, :, :]
        else:
            drift_corrected = disp

        return drift_corrected

    def smoothed_timesteps(self, nsteps, min_obs):
        """
        Calculate the smoothed timesteps to be used.

        Args:
            nsteps (:py:attr:`int`): Number of time steps.
        min_obs (:py:attr:`int`): Minimum number of observations to have before including in the MSD vs dt calculation. E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times.
                has 10 diffusing atoms, and min_obs = 30, the MSD vs dt will
                be calculated up to dt = total_run_time / 3, so that each
                diffusing atom is measured at least 3 uncorrelated times.

        Returns:
            :py:attr:`array_like`: Smoothed timesteps.
        """
        min_dt = int(1000 / (self.step_skip * self.time_step))
        max_dt = min(len(self.indices) * nsteps // min_obs, nsteps)
        if min_dt == 0:
            min_dt = 1
        if min_dt >= max_dt:
            raise ValueError('Not enough data to calculate diffusivity')
        timesteps = np.arange(
            min_dt,
            max_dt,
            max(int((max_dt - min_dt) / 200), 1),
        )
        return timesteps

    def get_disps(self, timesteps, drift_corrected):
        """
        Calculate the mean-squared displacement

        Args:
            timesteps (:py:attr:`array_like`): Smoothed timesteps.
            drift_corrected (:py:attr:`array_like`): Drift of framework corrected disp.

        Returns:
            :py:attr:`tuple`: Containing:
                - :py:attr:`array_like`: Time step intervals.
                - :py:attr:`array_like`: Mean-squared displacement.
                - :py:attr:`array_like`: Raw squared displacement.
        """
        delta_t = timesteps * self.time_step * self.step_skip
        disp_store = []
        disp_3d = []
        for timestep in tqdm(timesteps, desc='Getting Displacements'):
            disp = np.subtract(
                drift_corrected[:, timestep:, :],
                drift_corrected[:, :-timestep, :]
            )
            disp_3d.append(disp)
            disp_store.append(np.sum(disp, axis=2)[self.indices])

        return delta_t, disp_store, disp_3d


class MDAnalysisParser(PymatgenParser):
    """
    A parser that consumes an MDAnalysis.Universe object.

    Attributes:
        universe (:py:class:`MDAnalysis.core.universe.Universe`): The MDAnalysis object of interest.

    Args:
        universe (:py:class:`MDAnalysis.core.universe.Universe`): The MDAnalysis object of interest.
        specie (:py:attr:`str`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        time_step (:py:attr:`float`): Time step between measurements.
        step_step (:py:attr:`int`): Sampling freqency of the displacements (time_step is multiplied by this number to get the real time between measurements).
        min_obs (:py:attr:`int`, optional): Minimum number of observations to have before including in the MSD vs dt calculation. E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times. Defaults to :py:attr:`30`.
        sub_sample_atoms (:py:attr:`float`, optional): Fraction of atoms to be used. Defaults to :py:attr:`1` where all atoms are used.  
    """
    def __init__(self, universe, specie, time_step, step_skip, min_obs=30, sub_sample_atoms=1):
        
        self.universe = universe
        structures = []
        potential_indices = np.where(self.universe.atoms.types == str(pt.elements.symbol(specie).number))[0]
        atoms_indices = np.random.choice(potential_indices, size=int(potential_indices.size*sub_sample_atoms), replace=False)
        for t in tqdm(self.universe.trajectory, desc='Reading Trajectory'):
            structures.append(
                Structure(Lattice.from_parameters(*t.dimensions),
                        self.universe.atoms.types[atoms_indices],
                        self.universe.atoms.positions[atoms_indices],
                        coords_are_cartesian=True)
            )
        super().__init__(structures, specie, time_step, step_skip, min_obs=min_obs)



def _get_structure_and_disp(structures):
    """
    Obtain the initial structure and displacement from a Xdatcar file

    Args:
        structures (:py:attr:`list` or :py:class`pymatgen.core.structure.Structure`): Structures ordered in sequence of run. 
            
    Returns:
        :py:class`pymatgen.core.structure.Structure`: Initial structure.
        :py:attr:`array_like`: Numpy array of with shape [site, time step, axis].
    """
    coords, latt = [], []
    for i, struct in enumerate(structures):
        if i == 0:
            structure = struct
        coords.append(np.array(struct.frac_coords)[:, None])
        latt.append(struct.lattice.matrix)
    coords.insert(0, coords[0])
    latt.insert(0, latt[0])

    coords = np.concatenate(coords, axis=1)
    d_coords = coords[:, 1:] - coords[:, :-1]
    d_coords = d_coords - np.round(d_coords)
    f_disp = np.cumsum(d_coords, axis=1)
    c_disp = []
    for i in f_disp:
        c_disp.append([np.dot(d, m) for d, m in zip(i, latt[1:])])
    disp = np.array(c_disp)

    # If is NVT-AIMD, clear lattice data.
    if np.array_equal(latt[0], latt[-1]):
        latt = np.array([latt[0]])
    else:
        latt = np.array(latt)

    return structure, disp
