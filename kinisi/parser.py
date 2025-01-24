"""
Parser functions, including implementation for :py:mod:`pymatgen` compatible VASP files and :py:mod:`MDAnalysis`
compatible trajectories.

This parser borrows heavily from the :py:class:`pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer` class,
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
import warnings


class Parser:
    """
    The base class for parsing.

    :param disp: Displacements of atoms with the shape [site, time step, axis].
    :param indices: Indices for the atoms in the trajectory used in the diffusion calculation.
    :param drift_indices: Indices for the atoms in the trajectory that should not be used in the diffusion
        calculation, instead being used to correct for framework drift.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param min_dt: Minimum time interval to be evaluated, in the simulation units. Optional, defaults to the
        produce of :py:attr:`time_step` and :py:attr:`step_skip`.
    :param max_dt: Maximum time interval to be evaluated, in the simulation units. Optional, defaults to the
        length of the simulation.
    :param n_steps: Number of steps to be used in the time interval function. Optional, defaults to :py:attr:`100`
        unless this is fewer than the total number of steps in the trajectory when it defaults to this number.
    :param spacing: The spacing of the steps that define the time interval, can be either :py:attr:`'linear'` or
        :py:attr:`'logarithmic'`. If :py:attr:`'logarithmic'` the number of steps will be less than or equal
        to that in the :py:attr:`n_steps` argument, such that all values are unique. Optional, defaults to
        :py:attr:`linear`.
    :param sampling: The ways that the time-windows are sampled. The options are :py:attr:`'single-origin'`
        or :py:attr:`'multi-origin'` with the former resulting in only one observation per atom per
        time-window and the latter giving the maximum number of trajectories. Optional, defaults
        to :py:attr:`'multi-origin'`.
    :param memory_limit: Upper limit in the amount of computer memory that the displacements can occupy in
        gigabytes (GB). Optional, defaults to :py:attr:`8.`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    """

    def __init__(self,
                 disp: np.ndarray,
                 indices: np.ndarray,
                 drift_indices: np.ndarray,
                 time_step: float,
                 step_skip: int,
                 min_dt: float = None,
                 max_dt: float = None,
                 n_steps: int = 100,
                 spacing: str = 'linear',
                 sampling: str = 'multi-origin',
                 memory_limit: float = 8.,
                 progress: bool = True):

        self.time_step = time_step
        self.step_skip = step_skip
        self.indices = indices
        self.drift_indices = drift_indices
        self.memory_limit = memory_limit
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.sampling = sampling
        self._volume = None

        drift_corrected = self.correct_drift(drift_indices, disp)
        self.dc = drift_corrected

        if self.max_dt is None:
            self.max_dt = drift_corrected.shape[1] * (self.step_skip * self.time_step)
        if self.min_dt is None:
            self.min_dt = self.step_skip * self.time_step
        min_dt_int = int(self.min_dt / (self.step_skip * self.time_step))
        if min_dt_int >= drift_corrected.shape[1]:
            raise ValueError('The passed min_dt is greater than or equal to the maximum simulation length.')
        if n_steps > (drift_corrected.shape[1] - min_dt_int) + 1:
            n_steps = int(drift_corrected.shape[1] - min_dt_int) + 1

        self.time_intervals = self.get_time_intervals(n_steps, spacing)

        self.delta_t, self.disp_3d, self._n_o = self.get_disps(self.time_intervals, drift_corrected, progress)

    @property
    def volume(self) -> float:
        """
        :return: Volume of system, in cubic angstrom.
        """
        return self._volume

    @staticmethod
    def get_disp(coords: List[np.ndarray], latt: List[np.ndarray], progress: bool = True) -> np.ndarray:
        """
        Calculate displacements with support for NPT simualtions, using the TOR scheme outlined in doi: 10.1021/acs.jctc.3c00308.
        Issues a warning about non-orthorhombic cells. 

        :param coords: Fractional coordinates for all atoms.
        :param latt: Lattice descriptions.

        :return: Numpy array of with shape [site, time step, axis] describing displacements.
        """
        latt = np.array(latt)
        off_diags = np.array([[False, True, True], [True, False, True], [True, True, False]])
        triclinic_test = np.tile(off_diags, (latt.shape[0], 1, 1))
        if np.any(latt[triclinic_test] != 0):
            warnings.warn(
                'Converting triclinic cell to orthorhombic this may have unexpected results. '
                'Triclinic a, b, c are not equilivalent to orthorhombic x, y, z.', UserWarning)

        coords = np.concatenate(coords, axis=1)  #change array shape and removes extra dim
        latt_inv = np.linalg.inv(latt)  #invert lattice vectors
        wrapped = np.einsum('ijk,jkl->ijl', coords, latt)  #apply lattice vectors to get cartisian coords
        wrapped_diff = np.diff(wrapped, axis=1)  #calculate difference in cart

        diff_diff = np.einsum('ijk,jkl->ijl', np.floor(np.einsum('ijk,jkl->ijl', wrapped_diff, latt_inv[1:]) + (1 / 2)),
                              latt[1:])  # calculated the correction needed for the change in cell dimensions
        unwrapped_disp = wrapped_diff - diff_diff

        return np.cumsum(unwrapped_disp, axis=1)

    @staticmethod
    def correct_drift(drift_indices: np.ndarray, disp: np.ndarray) -> np.ndarray:
        """
        Perform drift correction, such that the displacement is calculated normalised to any framework drift.

        :param drift_indices: Indices for the atoms in the trajectory that should not be used in the diffusion
            calculation.
        :param disp: Numpy array of with shape [site, time step, axis] that describes the displacements.

        :return: Displacements corrected to account for drift of a framework.
        """
        # drift corrected position
        if len(drift_indices) > 0:
            framework_disp = disp[drift_indices]
            drift_corrected = disp - np.average(framework_disp, axis=0)[None, :, :]
        else:
            drift_corrected = disp
        return drift_corrected

    def get_time_intervals(self, n_steps: int, spacing: str) -> np.ndarray:
        """
        Calculate the smoothed time intervals to be used.

        :param n_steps: Number of time steps.
        :param step_spacing:

        :return: Smoothed time intervals.
        """
        min_dt = int(self.min_dt / (self.step_skip * self.time_step))
        max_dt = int(self.max_dt / (self.step_skip * self.time_step))
        if min_dt == 0:
            min_dt = 1
        if spacing == 'linear':
            return np.linspace(min_dt, max_dt, n_steps, dtype=int)
        elif spacing == 'logarithmic':
            return np.unique(np.geomspace(min_dt, max_dt, n_steps, dtype=int))
        else:
            raise ValueError("Only linear or logarithmic spacing is allowed.")

    def get_disps(self,
                  time_intervals: np.ndarray,
                  drift_corrected: np.ndarray,
                  progress: bool = True) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Calculate the displacement at each time interval.

        :param time_intervals: Smoothed time intervals.
        :param drift_corrected: Drift of framework corrected disp.
        :param progress: Print progress bars to screen. Defaults to :py:attr:`True`.

        :return: Tuple containing: time step intervals and raw displacement.
        """
        delta_t = time_intervals * self.time_step * self.step_skip
        disp_3d = []
        n_samples = np.array([])
        if progress:
            iterator = tqdm(time_intervals, desc='Getting Displacements')
        else:
            iterator = time_intervals
        disp_mem = 0
        itemsize = drift_corrected.itemsize
        for i, time_interval in enumerate(iterator):
            disp_mem += np.prod(drift_corrected[self.indices, time_interval::].shape) * itemsize
            disp_mem += (len(self.indices) * drift_corrected.shape[-1]) * itemsize
        disp_mem *= 1e-9
        if disp_mem > self.memory_limit:
            raise MemoryError(f"The memory limit of this job is {self.memory_limit:.1e} GB but the "
                              f"displacement values will use {disp_mem:.1e} GB. Please either increase "
                              "the memory_limit parameter or descrease the sampling rate (see "
                              "https://kinisi.readthedocs.io/en/latest/memory_limit.html).")
        for i, time_interval in enumerate(iterator):
            if self.sampling == 'single-origin':
                disp = drift_corrected[self.indices, i:i + 1]
                if np.multiply(*disp[:, ::time_interval].shape[:2]) <= 1:
                    continue
                disp_3d.append(disp)
            elif self.sampling == 'multi-origin':
                disp = np.concatenate([
                    drift_corrected[self.indices, np.newaxis, time_interval - 1],
                    np.subtract(drift_corrected[self.indices, time_interval:],
                                drift_corrected[self.indices, :-time_interval])
                ],
                                      axis=1)
                if np.multiply(*disp[:, ::time_interval].shape[:2]) <= 1:
                    continue
                disp_3d.append(disp)
            else:
                raise ValueError(f"The sampling option of {self.sampling} is unrecognized, "
                                 "please use 'multi-origin' or 'single-origin'.")
            # n_samples = np.append(n_samples, np.multiply(*disp[:, ::time_interval].shape[:2]))
            n_samples = np.append(n_samples, disp.shape[0] * time_intervals[-1] / time_interval)
        return delta_t, disp_3d, n_samples


class ASEParser(Parser):
    """
    A parser for ASE Atoms objects
    
    :param atoms: Atoms ordered in sequence of run.
    :param specie: symbol to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling frequency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional, defaults
        to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum time interval to be evaluated, in the simulation units. Optional, defaults to the
        produce of :py:attr:`time_step` and :py:attr:`step_skip`.
    :param max_dt: Maximum time interval to be evaluated, in the simulation units. Optional, defaults to the
        length of the simulation.
    :param n_steps: Number of steps to be used in the time interval function. Optional, defaults to :py:attr:`100`
        unless this is fewer than the total number of steps in the trajectory when it defaults to this number.
    :param spacing: The spacing of the steps that define the time interval, can be either :py:attr:`'linear'` or
        :py:attr:`'logarithmic'`. If :py:attr:`'logarithmic'` the number of steps will be less than or equal
        to that in the :py:attr:`n_steps` argument, such that all values are unique. Optional, defaults to
        :py:attr:`linear`.
    :param sampling: The ways that the time-windows are sampled. The options are :py:attr:`'single-origin'`
        or :py:attr:`'multi-origin'` with the former resulting in only one observation per atom per
        time-window and the latter giving the maximum number of origins without sampling overlapping
        trajectories. Optional, defaults to :py:attr:`'multi-origin'`.
    :param memory_limit: Upper limit in the amount of computer memory that the displacements can occupy in
        gigabytes (GB). Optional, defaults to :py:attr:`8.`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    :param specie_indices: Optional, list of indices to calculate diffusivity for as a list of indices. Specie 
        must be set to None for this to function. Molecules can be specificed as a list of lists of indices.
        The inner lists must all be on the same length.
    :param masses: Optional, list of masses associated with the indices in specie_indices. Must be same shape as specie_indices.
    :param framework_indices: Optional, list of framework indices to be used to correct framework drift. If an empty list is passed no drift correction will be performed. 
    """

    def __init__(self,
                 atoms: List["ase.atoms.Atoms"],
                 specie: str,
                 time_step: float,
                 step_skip: int,
                 sub_sample_traj: int = 1,
                 min_dt: float = None,
                 max_dt: float = None,
                 n_steps: int = 100,
                 spacing: str = 'linear',
                 sampling: str = 'multi-origin',
                 memory_limit: float = 8.,
                 progress: bool = True,
                 specie_indices: List[int] = None,
                 masses: List[float] = None,
                 framework_indices: List[int] = None):

        structure, coords, latt = self.get_structure_coords_latt(atoms, sub_sample_traj, progress)

        if specie is not None:
            indices = self.get_indices(structure, specie, framework_indices)
        elif isinstance(specie_indices, (list, tuple)):
            if isinstance(specie_indices[0], (list, tuple)):
                coords, indices = _get_molecules(structure, coords, specie_indices, masses, framework_indices)
            else:
                indices = _get_framework(structure, specie_indices, framework_indices)
        else:
            raise TypeError('Unrecognized type for specie or specie_indices')

        self.coords_check = coords[0]

        super().__init__(self.get_disp(coords, latt, progress=progress), indices[0], indices[1], time_step, step_skip,
                         min_dt, max_dt, n_steps, spacing, sampling, memory_limit, progress)
        self._volume = structure.get_volume()

    @staticmethod
    def get_structure_coords_latt(
            atoms: List["ase.atoms.Atoms"],
            sub_sample_traj: int = 1,
            progress: bool = True) -> Tuple["ase.atoms.Atoms", List[np.ndarray], List[np.ndarray]]:
        """
        Obtain the initial structure and displacement from a :py:attr:`list` of :py:class`pymatgen.core.structure.Structure`.

        :param structures: Structures ordered in sequence of run.
        :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional, default is :py:attr:`1`.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.

        :return: Tuple containing: initial structure, fractional coordinates for all atoms, and lattice descriptions.
        """
        coords, latt = [], []
        first = True
        if progress:
            iterator = tqdm(atoms[::sub_sample_traj], desc='Reading Trajectory')
        else:
            iterator = atoms[::sub_sample_traj]
        for struct in iterator:
            if first:
                structure = struct
                first = False
            scaled_positions = struct.get_scaled_positions()
            coords.append(np.array(scaled_positions)[:, None])
            latt.append(struct.cell[:])

        coords.insert(0, coords[0])
        latt.insert(0, latt[0])
        return structure, coords, latt

    @staticmethod
    def get_indices(structure: "ase.atoms.Atoms", specie: "str",
                    framework_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine framework and non-framework indices for a :py:mod:`ase` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        :param framework_indices: Indices of framework to be used in drift correction. If set to None will return all indices that are not in indices.

        :returns: Tuple containing: indices for the atoms in the trajectory used in the calculation of the diffusion
            and indices of framework atoms.
        """
        indices = []
        drift_indices = []
        if not isinstance(specie, List):
            specie = [specie]
        for i, site in enumerate(structure):
            if site.symbol in specie:
                indices.append(i)
            else:
                drift_indices.append(i)

        if len(indices) == 0:
            raise ValueError("There are no species selected to calculate the mean-squared displacement of.")

        if isinstance(framework_indices, (list, tuple)):
            drift_indices = framework_indices

        return indices, drift_indices


class PymatgenParser(Parser):
    """
    A parser for pymatgen structures.

    :param structures: Structures ordered in sequence of run.
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional, defaults
        to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum time interval to be evaluated, in the simulation units. Optional, defaults to the
        produce of :py:attr:`time_step` and :py:attr:`step_skip`.
    :param max_dt: Maximum time interval to be evaluated, in the simulation units. Optional, defaults to the
        length of the simulation.
    :param n_steps: Number of steps to be used in the time interval function. Optional, defaults to :py:attr:`100`
        unless this is fewer than the total number of steps in the trajectory when it defaults to this number.
    :param spacing: The spacing of the steps that define the time interval, can be either :py:attr:`'linear'` or
        :py:attr:`'logarithmic'`. If :py:attr:`'logarithmic'` the number of steps will be less than or equal
        to that in the :py:attr:`n_steps` argument, such that all values are unique. Optional, defaults to
        :py:attr:`linear`.
    :param sampling: The ways that the time-windows are sampled. The options are :py:attr:`'single-origin'`
        or :py:attr:`'multi-origin'` with the former resulting in only one observation per atom per
        time-window and the latter giving the maximum number of origins without sampling overlapping
        trajectories. Optional, defaults to :py:attr:`'multi-origin'`.
    :param memory_limit: Upper limit in the amount of computer memory that the displacements can occupy in
        gigabytes (GB). Optional, defaults to :py:attr:`8.`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    :param specie_indices: Optional, list of indices to calculate diffusivity for as a list of indices. Specie 
        must be set to None for this to function. Molecules can be specificed as a list of lists of indices.
        The inner lists must all be on the same length.
    :param masses: Optional, list of masses associated with the indices in specie_indices. Must be same shape as specie_indices.
    :param framework_indices: Optional, list of framework indices to be used to correct framework drift. If an empty list is passed no drift correction will be performed.
    """

    def __init__(self,
                 structures: List["pymatgen.core.structure.Structure"],
                 specie: Union["pymatgen.core.periodic_table.Element", "pymatgen.core.periodic_table.Specie"],
                 time_step: float,
                 step_skip: int,
                 sub_sample_traj: int = 1,
                 min_dt: float = None,
                 max_dt: float = None,
                 n_steps: int = 100,
                 spacing: str = 'linear',
                 sampling: str = 'multi-origin',
                 memory_limit: float = 8.,
                 progress: bool = True,
                 specie_indices: List[int] = None,
                 masses: List[float] = None,
                 framework_indices: List[int] = None):

        structure, coords, latt = self.get_structure_coords_latt(structures, sub_sample_traj, progress)

        if specie is not None:
            indices = self.get_indices(structure, specie, framework_indices)
        elif isinstance(specie_indices, (list, tuple)):
            if isinstance(specie_indices[0], (list, tuple)):
                coords, indices = _get_molecules(structure, coords, specie_indices, masses, framework_indices)
            else:
                indices = _get_framework(structure, specie_indices, framework_indices)
        else:
            raise TypeError('Unrecognized type for specie or specie_indices')

        self.coords_check = coords[0]

        super().__init__(disp=self.get_disp(coords, latt, progress=progress),
                         indices=indices[0],
                         drift_indices=indices[1],
                         time_step=time_step,
                         step_skip=step_skip * sub_sample_traj,
                         min_dt=min_dt,
                         max_dt=max_dt,
                         n_steps=n_steps,
                         spacing=spacing,
                         sampling=sampling,
                         memory_limit=memory_limit,
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
    def get_indices(structure: "pymatgen.core.structure.Structure",
                    specie: Union["pymatgen.core.periodic_table.Element", "pymatgen.core.periodic_table.Specie",
                                  "pymatgen.core.periodic_table.Species", List["pymatgen.core.periodic_table.Element"],
                                  List["pymatgen.core.periodic_table.Specie"],
                                  List["pymatgen.core.periodic_table.Species"]],
                    framework_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine framework and non-framework indices for a :py:mod:`pymatgen` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        :param framework_indices: Indices of framework to be used in drift correction. If set to None will return all indices that are not in indices.

        :returns: Tuple containing: indices for the atoms in the trajectory used in the calculation of the diffusion
            and indices of framework atoms.
        """
        indices = []
        drift_indices = []
        if not isinstance(specie, List):
            specie = [specie]
        for i, site in enumerate(structure):
            if site.specie.__str__() in specie:
                indices.append(i)
            else:
                drift_indices.append(i)

        if len(indices) == 0:
            raise ValueError("There are no species selected to calculate the mean-squared displacement of.")

        if isinstance(framework_indices, (list, tuple)):
            drift_indices = framework_indices

        return indices, drift_indices


class MDAnalysisParser(Parser):
    """
    A parser that consumes an MDAnalysis.Universe object.

    :param universe: The MDAnalysis object of interest.
    :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
    :param time_step: Time step, in picoseconds, between steps in trajectory.
    :param step_skip: Sampling freqency of the trajectory (time_step is multiplied by this number to get the real
        time between output from the simulation file).
    :param sub_sample_atoms: The sampling rate to sample the atoms in the system. Optional, defaults
        to :py:attr:`1` where all atoms are used.
    :param sub_sample_traj: Multiple of the :py:attr:`time_step` to sub sample at. Optional,
        defaults to :py:attr:`1` where all timesteps are used.
    :param min_dt: Minimum time interval to be evaluated, in the simulation units. Optional, defaults to the
        produce of :py:attr:`time_step` and :py:attr:`step_skip`.
    :param max_dt: Maximum time interval to be evaluated, in the simulation units. Optional, defaults to the
        length of the simulation.
    :param n_steps: Number of steps to be used in the time interval function. Optional, defaults to :py:attr:`100`
        unless this is fewer than the total number of steps in the trajectory when it defaults to this number.
    :param spacing: The spacing of the steps that define the time interval, can be either :py:attr:`'linear'` or
        :py:attr:`'logarithmic'`. If :py:attr:`'logarithmic'` the number of steps will be less than or equal
        to that in the :py:attr:`n_steps` argument, such that all values are unique. Optional, defaults to
        :py:attr:`linear`.
    :param sampling: The ways that the time-windows are sampled. The options are :py:attr:`'single-origin'`
        or :py:attr:`'multi-origin'` with the former resulting in only one observation per atom per
        time-window and the latter giving the maximum number of origins without sampling overlapping
        trajectories. Optional, defaults to :py:attr:`'multi-origin'`.
    :param memory_limit: Upper limit in the amount of computer memory that the displacements can occupy in
        gigabytes (GB). Optional, defaults to :py:attr:`8.`.
    :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
    :param specie_indices: Optional, list of indices to calculate diffusivity for as a list of indices. Specie 
        must be set to None for this to function. Molecules can be specificed as a list of lists of indices.
        The inner lists must all be on the same length.
    :param masses: Optional, list of masses associated with the indices in specie_indices. Must be same shape as specie_indices.
    :param framework_indices: Optional, list of framework indices to be used to correct framework drift. If an empty list is passed no drift correction will be performed.
    """

    def __init__(self,
                 universe: "MDAnalysis.core.universe.Universe",
                 specie: str,
                 time_step: float,
                 step_skip: int,
                 sub_sample_atoms: int = 1,
                 sub_sample_traj: int = 1,
                 min_dt: float = None,
                 max_dt: float = None,
                 n_steps: int = 100,
                 spacing: str = 'linear',
                 sampling: str = 'multi-origin',
                 memory_limit: float = 8.,
                 progress: bool = True,
                 specie_indices: List[int] = None,
                 masses: List[float] = None,
                 framework_indices: List[int] = None):

        if sub_sample_atoms != 1 and specie_indices is not None:
            raise ValueError(
                'sub_sample_atom cannot be used with specie_indices. Please specify only inidices you wish to sample.')

        structure, coords, latt, volume = self.get_structure_coords_latt(universe, sub_sample_atoms, sub_sample_traj,
                                                                         progress)

        if specie is not None:
            indices = self.get_indices(structure, specie, framework_indices)
        elif isinstance(specie_indices, (list, tuple)):
            if isinstance(specie_indices[0], (list, tuple)):
                coords, indices = _get_molecules(
                    structure, coords, specie_indices, masses, framework_indices
                )  #Warning: This function changes the structure without changing the MDAnalysis object
            else:
                indices = _get_framework(structure, specie_indices, framework_indices)
        else:
            raise TypeError('Unrecognized type for specie or specie_indices')

        self.coords_check = coords[0]

        super().__init__(disp=self.get_disp(coords, latt, progress=progress),
                         indices=indices[0],
                         drift_indices=indices[1],
                         time_step=time_step,
                         step_skip=step_skip * sub_sample_traj,
                         min_dt=min_dt,
                         max_dt=max_dt,
                         n_steps=n_steps,
                         spacing=spacing,
                         sampling=sampling,
                         memory_limit=memory_limit,
                         progress=progress)
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
            lattice descriptions, and the cell volume.
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
            matrix = timestep.triclinic_dimensions
            inv_matrix = np.linalg.inv(matrix)
            coords.append(np.array(np.dot(universe.atoms[::sub_sample_atoms].positions, inv_matrix))[:, None])
            latt.append(matrix)
        coords.insert(0, coords[0])
        latt.insert(0, latt[0])
        return structure, coords, latt, volume

    @staticmethod
    def get_indices(structure: "MDAnalysis.universe.Universe", specie: str,
                    framework_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine framework and non-framework indices for an :py:mod:`MDAnalysis` compatible file.

        :param structure: Initial structure.
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        :param framework_indices: Indices of framework to be used in drift correction. If set to None will return all indices that are not specie.

        :return: Tuple containing: indices for the atoms in the trajectory used in the calculation of the
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

        if isinstance(framework_indices, (list, tuple)):
            drift_indices = framework_indices

        return indices, drift_indices


def _get_molecules(structure: "ase.atoms.Atoms" or "pymatgen.core.structure.Structure"
                   or "MDAnalysis.universe.Universe", coords: List[np.ndarray], indices: List[int], masses: List[float],
                   framework_indices) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Determine framework and non-framework indices for an :py:mod:`ase` or :py:mod:`pymatgen` or :py:mod:`MDAnalysis` compatible file when specie_indices are provided and contain multiple molecules. Warning: This function changes the structure without changing the object.
    Calculates the centre of mass of provided particle groups using the pseudo-centre of mass approach (paper to come).

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
        indices = np.array(indices) - 1
    except:
        raise ValueError('Molecules must be of same length')

    n_molecules = indices.shape[0]

    if isinstance(framework_indices, (list, tuple)):
        drift_indices = np.array(framework_indices) - 1
    else:
        for i, site in enumerate(structure):
            if i not in indices:
                drift_indices.append(i)

    if masses == None:
        weights = None
    elif len(masses) != indices.shape[-1]:
        raise ValueError('Masses must be the same length as a molecule')
    else:
        masses = np.array(masses)
        weights = masses

    sq_coords = np.squeeze(coords, axis=2)
    s_coords = sq_coords[:, indices]
    theta = s_coords * (2 * np.pi)
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi_bar = np.average(xi, axis=-2, weights=weights)
    zeta_bar = np.average(zeta, axis=-2, weights=weights)
    theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
    new_s_coords = theta_bar / (2 * np.pi)

    # Implementation of pseudo-centre of mass approach to centre of mass calculation (paper to come).
    pseudo_com_recentering = ((s_coords - (new_s_coords + 0.5)[:, :, np.newaxis]) % 1)
    com_pseudo_space = np.average(pseudo_com_recentering, weights=masses, axis=2)
    corrected_com = (com_pseudo_space + (new_s_coords + 0.5)) % 1

    new_coords = np.concatenate((corrected_com, sq_coords[:, drift_indices]), axis=1)
    new_indices = list(range(n_molecules))
    new_drift_indices = list(range(n_molecules, n_molecules + len(drift_indices)))

    if new_coords.shape[2] != 1:
        new_coords = np.expand_dims(new_coords, axis=2)

    return new_coords, (new_indices, new_drift_indices)


def _get_framework(structure: "ase.atoms.Atoms" or "pymatgen.core.structure.Structure"
                   or "MDAnalysis.universe.Universe", indices: List[int],
                   framework_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine the framework indices from an :py:mod:`ase` or :py:mod:`pymatgen` or :py:mod:`MDAnalysis` compatible file when indices are provided
    
    :param structure: Initial structure.
    :param indices: Indices for the atoms in the trajectory used in the calculation of the 
        diffusion.
    :param framework_indices: Indices of framework to be used in drift correction. If set to None will return all indices that are not in indices.
    
    :return: Tuple containing: indices for the atoms in the trajectory used in the calculation of the
        diffusion and indices of framework atoms. 
    """
    if isinstance(framework_indices, (list, tuple)):
        drift_indices = np.array(framework_indices) - 1
    else:
        drift_indices = []

    for i, site in enumerate(structure):
        if i not in indices:
            drift_indices.append(i)

    return indices, drift_indices
