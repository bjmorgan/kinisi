"""
Parser functions, including implementation for :py:mod:`pymatgen` compatible VASP files and :py:mod:`MDAnalysis`
compatible trajectories.

This parser borrows heavily from the :py:class`pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer` class, 
originally authored by Will Richards (wrichard@mit.edu) and Shyue Ping Ong.
We include this statement to note that we make no claim to authorship of that code and make no attack on the
original authors.

In fact, we love pymatgen!
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import List, Tuple, Union
import numpy as np
from tqdm import tqdm


class Parser:
    """
    The base class for parsing.

    :param disp: Displacements of atoms with the shape [site, time step, axis].
    :param indices: Indices for the atoms in the trajectory used in the diffusion calculation.
    :param framework_indices: Indices for the atoms in the trajectory that should not be used in the diffusion
        calculation.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param min_obs: Minimum number of observations of an atom before including it in the MSD vs dt calculation.
        E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up
        to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated times.
        Optional, defaults to :py:attr:`30`.
    :param min_dt: Minimum timestep to be evaluated, in the simulation units. Optional, defaults to :py:attr:`100`.
    :param ndelta_t: The number of :py:attr:`delta_t` values to calculate the MSD over. Optional,
        defaults to :py:attr:`75`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    """
    def __init__(self,
                 disp: np.ndarray,
                 indices: np.ndarray,
                 framework_indices: np.ndarray,
                 time_step: float,
                 step_skip: int,
                 min_obs: int = 30,
                 min_dt: int = 1,
                 ndelta_t: int = 75,
                 progress: bool = True):
        self.time_step = time_step
        self.step_skip = step_skip
        self.indices = indices
        self.min_dt = min_dt
        self.ndelta_t = ndelta_t
        self._volume = None

        drift_corrected = self.correct_drift(framework_indices, disp)

        nsteps = drift_corrected.shape[1]

        timesteps = self.smoothed_timesteps(nsteps, min_obs, indices)

        self.delta_t, self.disp_3d = self.get_disps(timesteps, drift_corrected, progress)

    @property
    def volume(self) -> float:
        """
        :return: Volume of system, in cubic angstrom.
        """
        return self._volume

    @staticmethod
    def get_disp(coords: List[np.ndarray], latt: List[np.ndarray]) -> np.ndarray:
        """
        Calculate displacements.

        :param coords: Fractional coordinates for all atoms.
        :param latt: Lattice descriptions.

        :return: Numpy array of with shape [site, time step, axis] describing displacements.
        """
        coords = np.concatenate(coords, axis=1)
        d_coords = coords[:, 1:] - coords[:, :-1]
        d_coords = d_coords - np.round(d_coords)
        f_disp = np.cumsum(d_coords, axis=1)
        c_disp = []
        for i in f_disp:
            c_disp.append([np.dot(d, m) for d, m in zip(i, latt[1:])])
        disp = np.array(c_disp)
        return disp

    @staticmethod
    def correct_drift(framework_indices: np.ndarray, disp: np.ndarray) -> np.ndarray:
        """
        Perform drift correction, such that the displacement is calculated normalised to any framework drift.

        :param framework_indices: Indices for the atoms in the trajectory that should not be used in the diffusion
            calculation.
        :param disp: Numpy array of with shape [site, time step, axis] that describes the displacements.

        :return: Displacements corrected to account for drift of a framework.
        """
        # drift corrected position
        if len(framework_indices) > 0:
            framework_disp = disp[framework_indices]
            drift_corrected = disp - np.average(framework_disp, axis=0)[None, :, :]
        else:
            drift_corrected = disp
        return drift_corrected

    def smoothed_timesteps(self, nsteps: int, min_obs: int, indices: np.ndarray) -> np.ndarray:
        """
        Calculate the smoothed timesteps to be used.

        :param nsteps: Number of time steps.
        :param min_obs: Minimum number of observations to have before including in the MSD vs dt calculation.
            E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated
            up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3
            uncorrelated times.
        :param indices: Indices for the atoms in the trajectory used in the calculation of the diffusion.

        :return: Smoothed timesteps.
        """
        min_dt = int(self.min_dt / (self.step_skip * self.time_step))
        max_dt = min(len(indices) * nsteps // min_obs, nsteps)
        if min_dt == 0:
            min_dt = 1
        if min_dt >= max_dt:
            raise ValueError('Not enough data to calculate diffusivity')
        timesteps = np.arange(min_dt, max_dt, max(int((max_dt - min_dt) / self.ndelta_t), 1))
        return timesteps

    def get_disps(self,
                  timesteps: np.ndarray,
                  drift_corrected: np.ndarray,
                  progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the displacement at each timestep.

        :param timesteps: Smoothed timesteps.
        :param drift_corrected: Drift of framework corrected disp.
        :param progress: Print progress bars to screen. Defaults to :py:attr:`True`.

        :return: Tuple containing: time step intervals and raw displacement.
        """
        delta_t = timesteps * self.time_step * self.step_skip
        disp_3d = []
        if progress:
            iterator = tqdm(timesteps, desc='Getting Displacements')
        else:
            iterator = timesteps
        for timestep in iterator:
            disp = np.subtract(drift_corrected[self.indices, timestep:, :],
                               drift_corrected[self.indices, :-timestep, :])
            disp_3d.append(disp)
        return delta_t, disp_3d


class PymatgenParser(Parser):
    """
    A parser for pymatgen structures.

    :param structures: Structures ordered in sequence of run.
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param min_obs: Minimum number of observations of an atom before including it in the MSD vs dt calculation.
        E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated up
        to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated
        times. Optional, defaults to :py:attr:`30`.
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional, defaults
        to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum timestep to be evaluated, in the simulation units. Optional, defaults to :py:attr:`100`.
    :param ndelta_t: The number of :py:attr:`delta_t` values to calculate the MSD over. Optional,
        defaults to :py:attr:`75`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    """
    def __init__(self,
                 structures: List["pymatgen.core.structure.Structure"],
                 specie: Union["pymatgen.core.periodic_table.Element", "pymatgen.core.periodic_table.Specie"],
                 time_step: float,
                 step_skip: int,
                 min_obs: int = 30,
                 sub_sample_traj: int = 1,
                 min_dt: float = 100,
                 ndelta_t: int = 75,
                 progress: bool = True):
        structure, coords, latt = self.get_structure_coords_latt(structures, sub_sample_traj, progress)

        indices = self.get_indices(structure, specie)

        super().__init__(self.get_disp(coords, latt),
                         indices[0],
                         indices[1],
                         time_step,
                         step_skip,
                         min_obs=min_obs,
                         min_dt=min_dt,
                         ndelta_t=ndelta_t,
                         progress=progress)
        self._volume = structure.volume
        self.delta_t *= 1e-3

    @staticmethod
    def get_structure_coords_latt(
            structures: List["pymatgen.core.structure.Structure"],
            sub_sample_traj: int = 1,
            progress: bool = True) -> Tuple["pymatgen.core.structure.Structure", List[np.ndarray], List[np.ndarray]]:
        """
        Obtain the initial structure and displacement from a :py:attr:`list`
        of :py:class`pymatgen.core.structure.Structure`.

        :param structures: Structures ordered in sequence of run.
        :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at.
            Optional, default is :py:attr:`1`.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.

        :return: Tuple containing: initial structure, fractional coordinates for all atoms,
            and lattice descriptions.
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

    @staticmethod
    def get_indices(
        structure: "pymatgen.core.structure.Structure", specie: Union["pymatgen.core.periodic_table.Element",
                                                                      "pymatgen.core.periodic_table.Specie"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine framework and non-framework indices for a :py:mod:`pymatgen` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.

        :returns: Tuple containing: indices for the atoms in the trajectory used in the calculation of the diffusion
            and indices of framework atoms.
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

    :param universe: The MDAnalysis object of interest.
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param min_obs: Minimum number of observations of an atom before including it in the MSD vs dt calculation.
        E.g. If a structure has 10 diffusing atoms, and :py:attr:`min_obs=30`, the MSD vs dt will be calculated
        up to :py:attr:`dt = total_run_time / 3`, so that each diffusing atom is measured at least 3 uncorrelated
        times. Optional, defaults to :py:attr:`30`.
    :param sub_sample_atoms: The sampling rate to sample the atoms in the system. Optional, defaults 
        to :py:attr:`1` where all atoms are used.
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional,
        defaults to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum timestep to be evaluated, in the simulation units. Optional,
        defaults to :py:attr:`100`.
    :param ndelta_t: The number of :py:attr:`delta_t` values to calculate the MSD over. Optional,
        defaults to :py:attr:`75`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    """
    def __init__(self,
                 universe: "MDAnalysis.core.universe.Universe",
                 specie: str,
                 time_step: float,
                 step_skip: int,
                 min_obs: int = 30,
                 sub_sample_atoms: int = 1,
                 sub_sample_traj: int = 1,
                 min_dt: float = 100,
                 ndelta_t: int = 75,
                 progress: bool = True):
        structure, coords, latt, volume = self.get_structure_coords_latt(universe, sub_sample_atoms, sub_sample_traj,
                                                                         progress)

        indices = self.get_indices(structure, specie)

        super().__init__(self.get_disp(coords, latt), indices[0], indices[1], time_step, step_skip * sub_sample_traj,
                         min_obs, min_dt, ndelta_t, progress)
        self._volume = volume

    @staticmethod
    def get_structure_coords_latt(
            universe: "MDAnalysis.core.universe.Universe",
            sub_sample_atoms: int = 1,
            sub_sample_traj: int = 1,
            progress: bool = True
    ) -> Tuple["MDAnalysis.core.groups.AtomGroup", List[np.ndarray], List[np.ndarray], float]:
        """
        Obtain the initial structure and displacement from a :py:class:`MDAnalysis.universe.Universe` file.

        :param universe: Universe for analysis.
        :param sub_sample_atoms: Frequency to sub sample the number of atoms. Optional, default is :py:attr:`1`.
        :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional,
            default is :py:attr:`1`.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.

        :return: Tuple containing: initial structure, fractional coordinates for all atoms,
            lattice descriptions, and the cell volume
        """
        coords, latt = [], []
        first = True
        if progress:
            iterator = tqdm(universe.trajectory[::sub_sample_traj], desc='Reading Trajectory')
        else:
            iterator = universe.trajectory[::sub_sample_traj]
        volume = 0
        for timestep in iterator:
            if first:
                structure = universe.atoms[::sub_sample_atoms]
                first = False
                volume = timestep.volume
            matrix = _get_matrix(timestep.dimensions)
            inv_matrix = np.linalg.inv(matrix)
            coords.append(np.array(np.dot(universe.atoms[::sub_sample_atoms].positions, inv_matrix))[:, None])
            latt.append(matrix)
        coords.insert(0, coords[0])
        latt.insert(0, latt[0])
        return structure, coords, latt, volume

    @staticmethod
    def get_indices(structure: "MDAnalysis.universe.Universe", specie: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine framework and non-framework indices for an :py:mod:`MDAnalysis` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.

        :return: Tuple containing: indices for the atoms in the trajectory used in the calculation of the
            diffusion and indices of framework atoms.
        """
        indices = []
        framework_indices = []
        if not isinstance(specie, list):
            specie = [specie]
        for i, site in enumerate(structure):
            if site.type in specie:
                indices.append(i)
            else:
                framework_indices.append(i)
        return indices, framework_indices


def _get_matrix(dimensions: np.ndarray) -> np.ndarray:
    """
    Determine the lattice matrix.

    :param dimensions: a, b, c, vectors and alpha, beta, gamma angles.

    :return: Lattice matrix
    """
    angles_r = np.radians(dimensions[3:])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta = np.sin(angles_r)[:2]

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = max(min(val, 1), -1)
    gamma_star = np.arccos(val)

    vector_a = [dimensions[0] * sin_beta, 0.0, dimensions[0] * cos_beta]
    vector_b = [
        -dimensions[1] * sin_alpha * np.cos(gamma_star), dimensions[1] * sin_alpha * np.sin(gamma_star),
        dimensions[1] * cos_alpha
    ]
    vector_c = [0.0, 0.0, float(dimensions[2])]

    return np.array([vector_a, vector_b, vector_c], dtype=np.float64).reshape((3, 3))
