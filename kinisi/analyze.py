"""
This module contains the API classes for :py:mod:`kinisi`.
It is anticipated that this is where the majority of interaction with the package will occur.
This module includes the :py:class:`kinisi.analyze.MSDAnalyzer` class for MSD analysis and the :py:class:`kinisi.analyze.DiffAnalyzer` class for diffusion analysis, these are both compatible with VASP Xdatcar output files and any MD trajectory that the :py:mod:`MDAnalysis` package can handle.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from kinisi import diffusion
from kinisi.parser import MDAnalysisParser, PymatgenParser


class Analyzer:
    """
    The :py:class:`kinisi.analyze.Analyzer` class manages the API to the MSDAnalyzer and DiffAnalyzer classes.

    Attributes:
        delta_t (:py:attr:`array_like`):  Timestep values.
        msd (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations.
        _diff (:py:class:`kinisi.diffusion.MSDBootstrap`): The :py:mod:`kinisi` bootstrap and diffusion object.

    Args:
        trajectory (:py:attr:`str` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:class:`MDAnalysis.core.Universe`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes, a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects ordered in sequence of run, or an :py:class:`MDAnalysis.core.universe.Universe` object.
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary.
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, for a trajectory file path to be read by :py:class:`pymatgen.io.vasp.Xdatcar` this should be :py:attr:`'Xdatcar'`, for multiple trajectory files that are of the same system (but simulated from a different starting random seed) to be read by :py:class:`pymatgen.io.vasp.Xdatcar` then :py:attr:`'IdenticalXdatcar'` should be used (this assumes that all files have the same number of steps and atoms), for a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects this should be :py:attr:`'structures'`, for a trajectory file path to be read by :py:mod:`MDAnalysis` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`, and for a n :py:class:`MDAnalysis.core.universe.Universe` object this should be :py:attr:`'universe'`. Defaults to :py:attr:`'Xdatcar'`.
    """
    def __init__(self, trajectory, parser_params, dtype='pymatgen'):  
        if 'pymatgen' in dtype:
            try:
                from pymatgen.io.vasp import Xdatcar
                from pymatgen.core.structure import Structure
            except ModuleNotFoundError:  # pragma: no cover
                raise ModuleNotFoundError("To use the Xdatcar file parsing, pymatgen must be installed.")  # pragma: no cover
            if isinstance(trajectory, list):   
                if isinstance(trajectory[0], Structure):
                    u = PymatgenParser(trajectory, **parser_params)
                    self.delta_t = u.delta_t
                    self.disp_3d = u.disp_3d
                elif 'identical' in dtype:
                    if isinstance(trajectory[0], Xdatcar):
                        u = [PymatgenParser(f.structures, **parser_params) for f in trajectory] 
                    elif isinstance(trajectory[0], str):
                        u = [PymatgenParser(Xdatcar(f).structures, **parser_params) for f in trajectory]
                    elif isinstance(trajectory[0], list):
                        u = [PymatgenParser(f, **parser_params) for f in trajectory]
                    self.delta_t = u[0].delta_t
                    self.disp_3d = self.stack_trajectories(u)
                elif 'consecutive' in dtype:
                    if isinstance(trajectory[0], Xdatcar):
                        structures = _flatten_list([x.structures for x in trajectory])
                    elif isinstance(trajectory[0], str):
                        trajectory_list = (Xdatcar(f) for f in trajectory)
                        structures = _flatten_list([x.structures for x in trajectory_list])
                    elif isinstance(trajectory[0], list):
                        structures = _flatten_list([x for x in trajectory])
                    u = PymatgenParser(structures, **parser_params)
                    self.delta_t = u.delta_t
                    self.disp_3d = u.disp_3d 
                else:
                    raise ValueError("The structure of the input could not be recognised, please consult the documentation.")
            elif isinstance(trajectory, Xdatcar):
                structures = trajectory.structures
                u = PymatgenParser(structures, **parser_params)
                self.delta_t = u.delta_t
                self.disp_3d = u.disp_3d
            elif isinstance(trajectory, str):
                structures = Xdatcar(trajectory).structures
                u = PymatgenParser(structures, **parser_params)
                self.delta_t = u.delta_t
                self.disp_3d = u.disp_3d
            else:
                raise ValueError("The structure of the input could not be recognised, please consult the documentation.")
        if 'mdanalysis' in dtype:
            try:
                import MDAnalysis as mda
            except ModuleNotFoundError:  # pragma: no cover
                raise ModuleNotFoundError("To use the MDAnalysis from file parsing, MDAnalysis must be installed.")  # pragma: no cover
            if not isinstance(trajectory, mda.core.universe.Universe):
                raise ValueError('To use the MDAnalysis input, the trajectory must be an MDAnalysis.Universe.')
            u = MDAnalysisParser(trajectory, **parser_params)
            self.delta_t = u.delta_t
            self.disp_3d = u.disp_3d

    @staticmethod
    def stack_trajectories(u):
        """
        If more than one trajectory is given, then they are stacked to give the appearence that there are additional atoms in the trajectory:

        Args:
            u (:py:attr:`list` of :py:attr:`kinisi.parser.Parser`): Results from the parsing of each trajectory
        
        Returns: 
            :py:attr:`list` of :py:attr:`array_like`: The stacked displacement list
        """
        joint_disp_3d = []
        for i in range(len(u[0].disp_3d)):
            disp = np.zeros((u[0].disp_3d[i].shape[0] * len(u), u[0].disp_3d[i].shape[1], u[0].disp_3d[i].shape[2]))
            disp[:u[0].disp_3d[i].shape[0]] = u[0].disp_3d[i]
            for j in range(1, len(u)):
                disp[u[0].disp_3d[i].shape[0] * j:u[0].disp_3d[i].shape[0] * (j + 1)] = u[j].disp_3d[i]
            joint_disp_3d.append(disp)
        return joint_disp_3d


class MSDAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.MSDAnalyzer` class evaluates the MSD of atoms in a material.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically accurate values for mean squared displacement uncertainty and covariance.

    Attributes:
        delta_t (:py:attr:`array_like`):  Timestep values.
        msd (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations.
        _diff (:py:class:`kinisi.diffusion.MSDBootstrap`): The :py:mod:`kinisi` bootstrap and diffusion object.

    Args:
        trajectory (:py:attr:`str` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:class:`MDAnalysis.core.Universe`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes, a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects ordered in sequence of run, or an :py:class:`MDAnalysis.core.universe.Universe` object.
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary.
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters.
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, for a trajectory file path to be read by :py:class:`pymatgen.io.vasp.Xdatcar` this should be :py:attr:`'Xdatcar'`, for multiple trajectory files that are of the same system (but simulated from a different starting random seed) to be read by :py:class:`pymatgen.io.vasp.Xdatcar` then :py:attr:`'IdenticalXdatcar'` should be used (this assumes that all files have the same number of steps and atoms), for a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects this should be :py:attr:`'structures'`, for a trajectory file path to be read by :py:mod:`MDAnalysis` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`, and for a n :py:class:`MDAnalysis.core.universe.Universe` object this should be :py:attr:`'universe'`. Defaults to :py:attr:`'Xdatcar'`.
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, dtype='Xdatcar'): 
        if bootstrap_params is None:
            bootstrap_params = {}
        super().__init__(trajectory, parser_params, dtype)
        self._diff = diffusion.MSDBootstrap(self.delta_t, self.disp_3d, **bootstrap_params)

    @property
    def dt(self):
        """
        Returns the timestep values that have been sampled.

        Returns:
            :py:attr:`array_like`: Timestep values.
        """
        return self._diff.dt

    @property
    def msd(self):
        """
        Returns MSD for the input trajectories. Note that this is the bootstrap sampled MSD, not the numerical average from the data.

        Returns:
            :py:attr:`array_like`: MSD values.
        """
        return self._diff.msd

    @property
    def msd_std(self):
        """
        Returns MSD standard deviations values for the input trajectories.

        Returns:
            :py:attr:`array_like`: Standard deviation values for MSD.
        """
        return self._diff.msd_std

    @property
    def dr(self):
        """
        Returns a list of :py:class:`uravu.distribution.Distribution` objects that describe the euclidian displacement at each :py:attr:`dt`.

        Return:
            :py:attr:`list` of :py:class:`uravu.distribution.Distribution`: euclidian displacements at each :py:attr:`dt`.
        """
        return self._diff.euclidian_displacements

    @property
    def ngp_max(self):
        """
        Returns the position in dt where the non-Gaussian parameter is maximised.

        Return:
            :py:attr:`float`: dt where NGP is max.
        """
        return self.dt[self._diff.ngp.argmax()]

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


class DiffAnalyzer(MSDAnalyzer):
    """
    The :py:class:`kinisi.analyze.DiffAnalyzer` class performs analysis of diffusion relationships in materials.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically accurate values for mean squared displacement uncertainty and covariance.
    The time-dependence of the MSD is then modelled in a generalised least squares fashion to obtain the diffusion coefficient and offset using Markov chain Monte Carlo maximum likelihood sampling.

    Args:
        trajectory (:py:attr:`str` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:class:`MDAnalysis.core.Universe`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes, a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects ordered in sequence of run, or an :py:class:`MDAnalysis.core.universe.Universe` object.
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary.
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters.
        diffusion_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters.
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, for a trajectory file path to be read by :py:class:`pymatgen.io.vasp.Xdatcar` this should be :py:attr:`'Xdatcar'`, for multiple trajectory files that are of the same system (but simulated from a different starting random seed) to be read by :py:class:`pymatgen.io.vasp.Xdatcar` then :py:attr:`'IdenticalXdatcar'` should be used (this assumes that all files have the same number of steps and atoms), for a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects this should be :py:attr:`'structures'`, for a trajectory file path to be read by :py:mod:`MDAnalysis` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`, and for a n :py:class:`MDAnalysis.core.universe.Universe` object this should be :py:attr:`'universe'`. Defaults to :py:attr:`'Xdatcar'`.
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, diffusion_params=None, dtype='Xdatcar'): 
        if bootstrap_params is None:
            bootstrap_params = {}
        if diffusion_params is None:
            diffusion_params = {}
        super().__init__(trajectory, parser_params, bootstrap_params, dtype)
        # self._diff = diffusion.MSDBootstrap(self.delta_t, self.disp_3d, **bootstrap_params)
        self._diff.diffusion(**diffusion_params)


def _flatten_list(this_list):
    """
    Flatten nested lists.

    Args:
        this_list (:py:attr:`list`): List to be flattened.

    Returns:
        :py:attr:`list`: Flattened list.
    """
    return [item for sublist in this_list for item in sublist]
