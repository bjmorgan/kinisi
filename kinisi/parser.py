"""
Parser functions, including implementation for :py:mod:`pymatgen` compatible VASP files and :py:mod:`MDAnalysis` compatible trajectories.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0902,R0913

# This parser borrows heavily from the pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer class, originally authored by Will Richards (wrichard@mit.edu) and Shyue Ping Ong. We include this statement to not that we make no claim to authorship of that code and make no attack on the original authors.

# In fact, we love pymatgen!

import numpy as np
from pymatgen.analysis.diffusion_analyzer import get_conversion_factor
from pymatgen.core.lattice import Lattice
from pymatgen.core import Structure
import periodictable as pt
from tqdm import tqdm


class Parser:
    """
    The base class for parsing.
    
    Attributes:
        time_step (:py:attr:`float`): Time step between measurements.
        step_step (:py:attr:`int`): Sampling freqency of the displacements (time_step is multiplied by this number to get the real time between measurements).
        indices (:py:attr:`array_like`): Indices for the atoms in the trajectory used in the calculation of the diffusion.
        delta_t (:py:attr:`array_like`):  Timestep values.
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): Each element in the :py:attr:`list` has the axes [atom, displacement observation, dimension] and there is one element for each delta_t value. *Note: it is necessary to use a :py:attr:`list` of :py:attr:`array_like` as the number of observations is not necessary the same at each timestep point*.
        min_dt (:py:attr:`float`, optional): Minimum timestep to be evaluated.
    
    Args:
        disp (:py:attr:`array_like`): Displacements of atoms.
        structures (:py:attr:`list` of :py:class`pymatgen.core.structure.Structure`): Structures ordered in sequence of run. 
        specie (:py:class:`pymatgen.core.periodic_table.Element` or :py:class:`pymatgen.core.periodic_table.Specie`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        time_step (:py:attr:`float`): Time step between measurements.
        step_step (:py:attr:`int`): Sampling freqency of the displacements (time_step is multiplied by this number to get the real time between measurements).
        min_obs (:py:attr:`int`, optional): Minimum number of observations to have before including in the MSD vs dt calculation. E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times. Defaults to :py:attr:`30`.    
        min_dt (:py:attr:`float`, optional): Minimum timestep to be evaluated. Defaults to :py:attr:`100`.
        progress (:py:attr:`bool`, optional): Print progress bars to screen. Defaults to :py:attr:`True`.
    """
    def __init__(self, disp, indices, framework_indices, time_step, step_skip, min_obs=30, min_dt=100, progress=True):
        self.time_step = time_step
        self.step_skip = step_skip
        self.indices = indices
        self.min_dt = min_dt

        drift_corrected = self.correct_drift(framework_indices, disp)

        nsteps = drift_corrected.shape[1]

        timesteps = self.smoothed_timesteps(nsteps, min_obs, indices)

        self.delta_t, self.disp_3d = self.get_disps(
            timesteps, drift_corrected, progress)

    def smoothed_timesteps(self, nsteps, min_obs, indices):
        """
        Calculate the smoothed timesteps to be used.

        Args:
            nsteps (:py:attr:`int`): Number of time steps.
            min_obs (:py:attr:`int`): Minimum number of observations to have before including in the MSD vs dt calculation. E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times.
            indices (:py:attr:`array_like`): Indices for the atoms in the trajectory used in the calculation of the diffusion.

        Returns:
            :py:attr:`array_like`: Smoothed timesteps.
        """
        min_dt = int(self.min_dt / (self.step_skip * self.time_step))
        max_dt = min(len(indices) * nsteps // min_obs, nsteps)
        if min_dt == 0:
            min_dt = 1
        if min_dt >= max_dt:
            raise ValueError('Not enough data to calculate diffusivity')
        timesteps = np.arange(min_dt, max_dt, max(int((max_dt - min_dt) / 200), 1))
        return timesteps

    def get_disps(self, timesteps, drift_corrected, progress=True):
        """
        Calculate the mean-squared displacement

        Args:
            timesteps (:py:attr:`array_like`): Smoothed timesteps.
            drift_corrected (:py:attr:`array_like`): Drift of framework corrected disp.
            progress (:py:attr:`bool`, optional): Print progress bars to screen. Defaults to :py:attr:`True`.

        Returns:
            :py:attr:`tuple`: Containing:
                - :py:attr:`array_like`: Time step intervals.
                - :py:attr:`array_like`: Raw squared displacement.
        """
        delta_t = timesteps * self.time_step * self.step_skip
        disp_3d = []
        if progress:
            iterator = tqdm(timesteps, desc='Getting Displacements')
        else:
            iterator = timesteps
        for timestep in iterator:
            disp = np.subtract(drift_corrected[self.indices, timestep:, :], drift_corrected[self.indices, :-timestep, :])
            disp_3d.append(disp)
        return delta_t, disp_3d

    def correct_drift(self, framework_indices, disp):
        """        
        Perform drift correction

        Args:
            framework_indices (:py:attr:`array_like`): Indices of framework atoms, to correct against.
            disp (:py:attr:`array_like`): Numpy array of with shape [site, time step, axis].

        Returns:
            :py:attr:`array_like`: Drift of framework corrected disp.
        """
        # drift corrected position
        if len(framework_indices) > 0:
            framework_disp = disp[framework_indices]
            drift_corrected = disp - np.average(framework_disp, axis=0)[None, :, :]
        else:
            drift_corrected = disp

        return drift_corrected


class PymatgenParser(Parser):
    """
    A parser for pymatgen structures.
    
    Args:
        structures (:py:attr:`list` or :py:class`pymatgen.core.structure.Structure`): Structures ordered in sequence of run. 
        specie (:py:class:`pymatgen.core.periodic_table.Element` or :py:class:`pymatgen.core.periodic_table.Specie`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        time_step (:py:attr:`float`): Time step between measurements.
        step_step (:py:attr:`int`): Sampling frequency of the displacements (time_step is multiplied by this number to get the real time between measurements).
        sub_sample_traj (:py:attr:`float`, optional): Multiple of the :py:attr:`time_step` to sub sample at. Defaults to :py:attr:`1` where all timesteps are used. 
        min_obs (:py:attr:`int`, optional): Minimum number of observations to have before including in the MSD vs dt calculation. E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times. Defaults to :py:attr:`30`.    
        min_dt (:py:attr:`float`, optional): Minimum timestep to be evaluated. Defaults to :py:attr:`100`.
        progress (:py:attr:`bool`, optional): Print progress bars to screen. Defaults to :py:attr:`True`.
    """
    def __init__(self, structures, specie, time_step, step_skip, min_obs=30, sub_sample_traj=1, min_dt=100, progress=True):
        structure, coords, latt = _pmg_get_structure_coords_latt(structures, sub_sample_traj, progress)

        disp, latt = _get_disp(coords, latt)

        indices, framework_indices = self.get_indices(structure, specie)

        super().__init__(disp, indices, framework_indices, time_step, step_skip, min_obs, min_dt, progress)

    def get_indices(self, structure, specie):
        """
        Determine framework and non-framework indices

        Perform drift correction

        Args:
            structure (:py:class:`pymatgen.core.structure.Structure`): Initial structure.
            specie (:py:class:`pymatgen.core.periodic_table.Element` or :py:class:`pymatgen.core.periodic_table.Specie`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.

        Returns:
            :py:attr:`array_like`: Indices for the atoms in the trajectory used in the calculation of the diffusion.
            :py:attr:`array_like`: Indices of framework atoms.
        """
        indices = []
        framework_indices = []
        for i, site in enumerate(structure):
            if site.specie.symbol == specie:
                indices.append(i)
            else:
                framework_indices.append(i)
        return indices, framework_indices


class MDAnalysisParser(Parser):
    """
    A parser that consumes an MDAnalysis.Universe object.

    Args:
        universe (:py:class:`MDAnalysis.core.universe.Universe`): The MDAnalysis object of interest.
        specie (:py:attr:`str`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        time_step (:py:attr:`float`): Time step between measurements.
        step_step (:py:attr:`int`): Sampling freqency of the displacements (time_step is multiplied by this number to get the real time between measurements).
        sub_sample_traj (:py:attr:`float`, optional): Multiple of the :py:attr:`time_step` to sub sample at. Defaults to :py:attr:`1` where all timesteps are used. 
        min_obs (:py:attr:`int`, optional): Minimum number of observations to have before including in the MSD vs dt calculation. E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times. Defaults to :py:attr:`30`.
        min_dt (:py:attr:`float`, optional): Minimum timestep to be evaluated. Defaults to :py:attr:`100`.
        progress (:py:attr:`bool`, optional): Print progress bars to screen. Defaults to :py:attr:`True`.
    """
    def __init__(self, universe, specie, time_step, step_skip, min_obs=30, sub_sample_traj=1, min_dt=100, progress=True):
        structure, coords, latt = _mda_get_structure_coords_latt(universe, specie, sub_sample_traj, progress)

        disp, latt = _get_disp(coords, latt)

        indices, framework_indices = self.get_indices(structure, specie)
               
        super().__init__(disp, indices, framework_indices, time_step, step_skip * sub_sample_traj, min_obs, min_dt, progress)

    def get_indices(self, structure, specie):
        """
        Determine framework and non-framework indices

        Perform drift correction

        Args:
            structure (:py:class:`MDAnalysis.core.groups.AtomGroup`): Initial structure.
            specie (:py:class:`pymatgen.core.periodic_table.Element` or :py:class:`pymatgen.core.periodic_table.Specie`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.

        Returns:
            :py:attr:`array_like`: Indices for the atoms in the trajectory used in the calculation of the diffusion.
            :py:attr:`array_like`: Indices of framework atoms.
        """
        indices = []
        framework_indices = []
        for i, site in enumerate(structure):
            if site.type == str(pt.elements.symbol(specie).number):
                indices.append(i)
            else:
                framework_indices.append(i)
        return indices, framework_indices


def _mda_get_structure_coords_latt(universe, specie, sub_sample_traj=1, progress=True):
    """
    Obtain the initial structure and displacement from a :py:class:`MDAnalysis.universe.Universe` file

    Args:
        universe (:py:class:MDAnalysis.universe.Universe): Universe for analysis.
        specie (:py:attr:`str`): Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`. 
        sub_sample_traj (:py:attr:`float`, optional): Multiple of the :py:attr:`time_step` to sub sample at. Default is :py:attr:`1`.
        progress (:py:attr:`bool`, optional): Print progress bars to screen. Defaults to :py:attr:`True`.
            
    Returns:
        :py:class`pymatgen.core.structure.Structure`: Initial structure.
        :py:attr:`list` of :py:attr:`array_like`: Fractional coordinates for all atoms.
        :py:attr:`list` of :py:attr:`array_like`: Lattice descriptions.
    """
    coords, latt = [], []
    first = True
    if progress:
        iterator = tqdm(universe.trajectory[::sub_sample_traj], desc='Reading Trajectory')
    else:
        iterator = universe.trajectory[::sub_sample_traj]
    for timestep in iterator:
        if first:
            structure = universe.atoms
            first = False
        matrix = get_matrix(timestep.dimensions)
        inv_matrix = np.linalg.inv(matrix)
        coords.append(np.array(np.dot(universe.atoms.positions, inv_matrix))[:, None])
        latt.append(matrix)
    coords.insert(0, coords[0])
    latt.insert(0, latt[0])
    return structure, coords, latt


def get_matrix(dimensions):
    """
    Determine the lattice matrix. 

    Args:
        dimensions (:py:attr:`array_like`): a, b, c, vectors and alpha, beta, gamma angles.

    Returns:
        :py:attr:`array_like`: Lattice matrix 
    """
    a, b, c, alpha, beta, gamma = dimensions

    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = max(min(val, 1), -1)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [-b * sin_alpha * np.cos(gamma_star), b * sin_alpha * np.sin(gamma_star), b * cos_alpha]
    vector_c = [0.0, 0.0, float(c)]

    return np.array([vector_a, vector_b, vector_c], dtype=np.float64).reshape((3, 3))


def _pmg_get_structure_coords_latt(structures, sub_sample_traj=1, progress=True):
    """
    Obtain the initial structure and displacement from a :py:attr:`list` of :py:class`pymatgen.core.structure.Structure`.

    Args:
        structures (:py:attr:`list` of :py:class`pymatgen.core.structure.Structure`): Structures ordered in sequence of run. 
        sub_sample_traj (:py:attr:`float`, optional): Multiple of the :py:attr:`time_step` to sub sample at. Default is :py:attr:`1`.
        progress (:py:attr:`bool`, optional): Print progress bars to screen. Defaults to :py:attr:`True`.
            
    Returns:
        :py:class`pymatgen.core.structure.Structure`: Initial structure.
        :py:attr:`list` of :py:attr:`array_like`: Fractional coordinates for all atoms.
        :py:attr:`list` of :py:attr:`array_like`: Lattice descriptions.
    """
    coords, latt = [], []
    first = True
    if progress:
        iterator = tqdm(structures[::sub_sample_traj], desc='Reading Trajectory')
    else:
        iterator = structures[::sub_sample_traj]
    for struct in iterator:
        if first:
            structure = struct
            first = False
        coords.append(np.array(struct.frac_coords)[:, None])
        latt.append(struct.lattice.matrix)
    coords.insert(0, coords[0])
    latt.insert(0, latt[0])

    return structure, coords, latt

def _get_disp(coords, latt):
    """
    Calculate displacements.

    Args:
        coords (:py:attr:`list` of :py:attr:`array_like`): Fractional coordinates for all atoms.
        latt (:py:attr:`list` of :py:attr:`array_like`): Lattice descriptions.

    Returns:
        :py:attr:`array_like`: Numpy array of with shape [site, time step, axis] describing displacements.
    """
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

    return disp, latt
