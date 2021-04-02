"""
This module contains the API classes for :py:mod:`kinisi`.
It is anticipated that this is where the majority of interaction with the package will occur.
This module includes the :py:class:`~kinisi.analyze.DiffAnalyzer` class for diffusion analysis, which is compatible with both VASP Xdatcar output files and any MD trajectory that the :py:mod:`MDAnalysis` package can handle.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
from uravu.distribution import Distribution
from kinisi import diffusion
from kinisi.parser import MDAnalysisParser, PymatgenParser


class MSDAnalyzer:
    """
    The :py:class:`kinisi.analyze.MSDAnalyzer` class evaluates the MSD of atoms in a material.
    This is achieved through the application of a block bootstrapping methodology to obtain the most statistically accurate values for mean squared displacement and the associated uncertainty.

    Attributes:
        dt (:py:attr:`array_like`):  Timestep values.
        distributions (:py:attr:`list` or :py:class:`Distribution`): The distributions describing the MSD at each timestep.
        msd_observed (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations.
        msd_sampled (:py:attr:`array_like`): The population mean-squared displacements, found from the bootstrap resampling of the observations.
        msd_sampled_err (:py:attr:`array_like`): The two-dimensional uncertainties, at the given confidence interval, found from the bootstrap resampling of the observations.
        _diff (:py:class:`kinisi.diffusion.MSDBootstrap`): The :py:mod:`kinisi` bootstrap and diffusion object.

    Args:
        trajectory (:py:attr:`str` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:class:`MDAnalysis.core.Universe`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes, a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects ordered in sequence of run, or an :py:class:`MDAnalysis.core.universe.Universe` object.
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary.
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters.
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, for a trajectory file path to be read by :py:class:`pymatgen.io.vasp.Xdatcar` this should be :py:attr:`'Xdatcar'`, for multiple trajectory files that are of the same system (but simulated from a different starting random seed) to be read by :py:class:`pymatgen.io.vasp.Xdatcar` then :py:attr:`'IdenticalXdatcar'` should be used (this assumes that all files have the same number of steps and atoms), for a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects this should be :py:attr:`'structures'`, for a trajectory file path to be read by :py:mod:`MDAnalysis` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`, and for a n :py:class:`MDAnalysis.core.universe.Universe` object this should be :py:attr:`'universe'`. Defaults to :py:attr:`'Xdatcar'`.
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, dtype='Xdatcar'):  # pragma: no cover
        if bootstrap_params is None:
            bootstrap_params = {}
        if dtype is 'Xdatcar':
            try:
                from pymatgen.io.vasp import Xdatcar
            except ModuleNotFoundError:
                raise ModuleNotFoundError("To use the Xdatcar file parsing, pymatgen must be installed.")
            if isinstance(trajectory, list):
                trajectory_list = (Xdatcar(f) for f in trajectory)
                structures = _flatten_list([x.structures for x in trajectory_list])
            else:
                xd = Xdatcar(trajectory)
                structures = xd.structures
            u = PymatgenParser(structures, **parser_params)
            self.first_structure = structures[0]
            dt = u.delta_t
            disp_3d = u.disp_3d
        elif dtype is 'IdenticalXdatcar':
            try:
                from pymatgen.io.vasp import Xdatcar
            except ModuleNotFoundError:
                raise ModuleNotFoundError("To use the Xdatcar file parsing, pymatgen must be installed.")
            u = [PymatgenParser(Xdatcar(f).structures, **parser_params) for f in trajectory]
            joint_disp_3d = []
            for i in range(len(u[0].disp_3d)):
                disp = np.zeros((u[0].disp_3d[i].shape[0] * len(u), u[0].disp_3d[i].shape[1], u[0].disp_3d[i].shape[2]))
                disp[:u[0].disp_3d[i].shape[0]] = u[0].disp_3d[i]
                for j in range(1, len(u)):
                    disp[u[0].disp_3d[i].shape[0] * j:u[0].disp_3d[i].shape[0] * (j+1)] = u[j].disp_3d[i]
                joint_disp_3d.append(disp)
            dt = u[0].delta_t
            disp_3d = joint_disp_3d
        elif dtype is 'structures':
            u = PymatgenParser(trajectory, **parser_params)
            self.first_structure = trajectory[0]
            dt = u.delta_t
            disp_3d = u.disp_3d
        elif dtype == 'universe':
            u = MDAnalysisParser(trajectory, **parser_params)
            dt = u.delta_t
            disp_3d = u.disp_3d
        else:
            try:
                import MDAnalysis as mda
            except ModuleNotFoundError:
                raise ModuleNotFoundError("To use the MDAnalysis from file parsing, MDAnalysis must be installed.")
            universe = mda.Universe(*trajectory, format=dtype)
            u = MDAnalysisParser(universe, **parser_params)
            dt = u.delta_t
            disp_3d = u.disp_3d

        self._diff = diffusion.MSDBootstrap(dt, disp_3d, **bootstrap_params)

        self.dt = self._diff.dt
        self.msd_sampled = self._diff.msd_sampled
        self.msd_sampled_err = self._diff.msd_sampled_err
        self.msd_observed = self._diff.msd_observed
        self.distributions = self._diff.distributions

    @property
    def msd(self):
        """
        Returns MSD for the input trajectories. Note that this is the bootstrap sampled MSD, not the numerical average from the data.

        Returns:
            :py:attr:`array_like`: MSD values.
        """
        return self.msd_sampled

    @property
    def msd_err(self):
        """
        Returns MSD uncertainty for the input trajectories.

        Returns:
            :py:attr:`array_like`: A lower and upper uncertainty, at a single standard deviation, of the mean squared displacement values.
        """
        return self.msd_sampled_err


class DiffAnalyzer(MSDAnalyzer):
    """
    The :py:class:`kinisi.analyze.DiffAnalyzer` class performs analysis of diffusion relationships in materials.
    This is achieved through the application of a block bootstrapping methodology to obtain the most statistically accurate values for mean squared displacement and the associated uncertainty.
    The time-scale dependence of the MSD is then modeled with a straight line Einstein relationship, and Markov chain Monte Carlo is used to quantify inverse uncertainties for this model.

    Args:
        trajectory (:py:attr:`str` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:class:`MDAnalysis.core.Universe`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes, a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects ordered in sequence of run, or an :py:class:`MDAnalysis.core.universe.Universe` object.
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary.
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters.
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, for a trajectory file path to be read by :py:class:`pymatgen.io.vasp.Xdatcar` this should be :py:attr:`'Xdatcar'`, for multiple trajectory files that are of the same system (but simulated from a different starting random seed) to be read by :py:class:`pymatgen.io.vasp.Xdatcar` then :py:attr:`'IdenticalXdatcar'` should be used (this assumes that all files have the same number of steps and atoms), for a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects this should be :py:attr:`'structures'`, for a trajectory file path to be read by :py:mod:`MDAnalysis` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`, and for a n :py:class:`MDAnalysis.core.universe.Universe` object this should be :py:attr:`'universe'`. Defaults to :py:attr:`'Xdatcar'`.
        n_samples_fit (:py:attr:`int`, optional): The number of samples in the random generator for the fitting of the linear relationship. Default is :py:attr:`10000`.
        fit_intercept (:py:attr:`bool`, optional): Should the intercept of the diffusion relationship be fit. Default is :py:attr:`True`.
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, dtype='Xdatcar', n_samples_fit=10000, fit_intercept=True):  # pragma: no cover
        super().__init__(trajectory, parser_params, bootstrap_params, dtype)
        self._diff.diffusion(n_samples_fit, fit_intercept)

    @property
    def D(self):
        """
        Diffusion coefficient.

        Returns:
            :py:class:`uravu.distribution.Distribution`: Diffusion coefficient.
        """
        return self._diff.diffusion_coefficient

    @property
    def D_offset(self):
        """
        Offset from abscissa.

        Returns:
            :py:class:`uravu.distribution.Distribution`: Abscissa offset.
        """
        return self._diff.intercept


def _flatten_list(this_list):
    """
    Flatten nested lists.

    Args:
        this_list (:py:attr:`list`): List to be flattened.

    Returns:
        :py:attr:`list`: Flattened list.
    """
    return [item for sublist in this_list for item in sublist]
