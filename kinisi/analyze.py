"""
This module contains the API classes for :py:mod:`kinisi`.
It is anticipated that this is where the majority of interaction with the package will occur.
This module includes the :py:class:`kinisi.analyze.DiffusionAnalyzer` class for MSD and diffusion analysis, the :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` class for TMSD and collective diffusion analysis, and the :py:class:`kinisi.analyze.ConductivityAnalyzer` class for MSCD and conductivity analysis, these are all compatible with VASP Xdatcar output files, pymatgen structures and any MD trajectory that the :py:mod:`MDAnalysis` package can handle.
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
        delta_t (:py:attr:`array_like`):  Timestep values
        disp_3d (:py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        volume (:py:attr:`float`): System volume

    Args:
        trajectory (:py:attr:`str` or :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:class:`MDAnalysis.core.Universe` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:attr:`list` of :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure`): The trajectory/ies to be analysed, supported trajectories are those from VASP (as file paths to be read, :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects, or a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects) or in the format of an :py:class:`MDAnalysis.core.Universe` (:py:attr:`dtype` should contain :py:attr:`'vasp'` or :py:attr:`'mdanalysis'` as appropriate). Additionally, for VASP trajectories, a list of identical starting points can be passed by using :py:attr:`dtype='identicalvasp'` and consecutive VASP trajectories using :py:attr:`dtype='consecutivevasp'`
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, see documentation for :py:attr:`trajectory` above. Defaults to :py:attr:`'vasp'`
    """
    def __init__(self, trajectory, parser_params, dtype='vasp'):
        if 'vasp' in dtype:
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
                    self.volume = u.volume
                elif 'identical' in dtype:
                    if isinstance(trajectory[0], Xdatcar):
                        u = [PymatgenParser(f.structures, **parser_params) for f in trajectory]
                    elif isinstance(trajectory[0], str):
                        u = [PymatgenParser(Xdatcar(f).structures, **parser_params) for f in trajectory]
                    elif isinstance(trajectory[0], list):
                        u = [PymatgenParser(f, **parser_params) for f in trajectory]
                    self.delta_t = u[0].delta_t
                    self.disp_3d = self.stack_trajectories(u)
                    self.volume = u[0].volume
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
                    self.volume = u.volume
                else:
                    raise ValueError("The structure of the input could not be recognised, please consult the documentation.")
            elif isinstance(trajectory, Xdatcar):
                structures = trajectory.structures
                u = PymatgenParser(structures, **parser_params)
                self.delta_t = u.delta_t
                self.disp_3d = u.disp_3d
                self.volume = u.volume
            elif isinstance(trajectory, str):
                structures = Xdatcar(trajectory).structures
                u = PymatgenParser(structures, **parser_params)
                self.delta_t = u.delta_t
                self.disp_3d = u.disp_3d
                self.volume = u.volume
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
            self.volume = u.volume

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

    @property
    def dr(self):
        """
        Returns a list of :py:class:`uravu.distribution.Distribution` objects that describe the euclidian displacement at each :py:attr:`dt`.

        Returns:
            :py:attr:`list` of :py:class:`uravu.distribution.Distribution`: euclidian displacements at each :py:attr:`dt`
        """
        return self._diff.euclidian_displacements

    @property
    def ngp_max(self):
        """
        Returns the position in dt where the non-Gaussian parameter is maximised.

        Returns:
            :py:attr:`float`: dt where NGP is max
        """
        return self.dt[self._diff.ngp.argmax()]

    @property
    def intercept(self):
        """
        Returns the distribution describing the intercept.

        Returns:
            :py:class:`uravu.distribution.Distribution`: Intercept
        """
        return self._diff.intercept


class DiffusionAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.DiffusionAnalyzer` class performs analysis of diffusion relationships in materials.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically accurate values for mean squared displacement uncertainty and covariance.
    The time-dependence of the MSD is then modelled in a generalised least squares fashion to obtain the diffusion coefficient and offset using Markov chain Monte Carlo maximum likelihood sampling.

    Args:
        trajectory (:py:attr:`str` or :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:class:`MDAnalysis.core.Universe` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:attr:`list` of :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure`): The trajectory/ies to be analysed, supported trajectories are those from VASP (as file paths to be read, :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects, or a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects) or in the format of an :py:class:`MDAnalysis.core.Universe` (:py:attr:`dtype` should contain :py:attr:`'vasp'` or :py:attr:`'mdanalysis'` as appropriate). Additionally, for VASP trajectories, a list of identical starting points can be passed by using :py:attr:`dtype='identicalvasp'` and consecutive VASP trajectories using :py:attr:`dtype='consecutivevasp'`
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, see documentation for :py:attr:`trajectory` above. Defaults to :py:attr:`'vasp'`
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, dtype='vasp'):
        if bootstrap_params is None:
            bootstrap_params = {}
        super().__init__(trajectory, parser_params, dtype)
        self._diff = diffusion.MSDBootstrap(self.delta_t, self.disp_3d, **bootstrap_params)

    def diffusion(self, diffusion_params=None):
        """
        Calculate the diffusion coefficicent using the bootstrap-GLS methodology.

        Args:
            diffusion_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters
        """
        if diffusion_params is None:
            diffusion_params = {}
        self._diff.diffusion(**diffusion_params)

    @property
    def dt(self):
        """
        Returns the timestep values that have been sampled.

        Returns:
            :py:attr:`array_like`: Timestep values
        """
        return self._diff.dt

    @property
    def msd(self):
        """
        Returns MSD for the input trajectories. Note that this is the bootstrap sampled MSD, not the numerical average from the data.

        Returns:
            :py:attr:`array_like`: MSD values
        """
        return self._diff.n

    @property
    def msd_std(self):
        """
        Returns MSD standard deviations values for the input trajectories.

        Returns:
            :py:attr:`array_like`: Standard deviation values for MSD
        """
        return self._diff.s

    @property
    def D(self):
        """
        Diffusion coefficient.

        Returns:
            :py:class:`uravu.distribution.Distribution`: Diffusion coefficient
        """
        return self._diff.diffusion_coefficient


class JumpDiffusionAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` class performs analysis of collective diffusion relationships in materials.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically accurate values for total mean squared displacement uncertainty and covariance.
    The time-dependence of the TMSD is then modelled in a generalised least squares fashion to obtain the jump diffusion coefficient and offset using Markov chain Monte Carlo maximum likelihood sampling.

    Args:
        trajectory (:py:attr:`str` or :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:class:`MDAnalysis.core.Universe` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:attr:`list` of :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure`): The trajectory/ies to be analysed, supported trajectories are those from VASP (as file paths to be read, :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects, or a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects) or in the format of an :py:class:`MDAnalysis.core.Universe` (:py:attr:`dtype` should contain :py:code:`vasp` or :py:code:`mdanalysis` as appropriate). Additionally, for VASP trajectories, a list of identical starting points can be passed by using :py:code:`dtype='identicalvasp'` and consecutive VASP trajectories using :py:code:`dtype='consecutivevasp'`
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, see documentation for :py:attr:`trajectory` above. Defaults to :py:attr:`'vasp'`
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, dtype='vasp'):
        if bootstrap_params is None:
            bootstrap_params = {}
        super().__init__(trajectory, parser_params, dtype)
        self._diff = diffusion.TMSDBootstrap(self.delta_t, self.disp_3d, **bootstrap_params)

    def jump_diffusion(self, jump_diffusion_params=None):
        """
        Calculate the jump diffusion coefficicent using the bootstrap-GLS methodology.

        Args:
            jump_diffusion_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.TMSDBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters
        """
        if jump_diffusion_params is None:
            jump_diffusion_params = {}
        self._diff.jump_diffusion(**jump_diffusion_params)

    @property
    def dt(self):
        """
        Returns the timestep values that have been sampled.

        Returns:
            :py:attr:`array_like`: Timestep values
        """
        return self._diff.dt

    @property
    def tmsd(self):
        """
        Returns TMSD for the input trajectories. Note that this is the bootstrap sampled MSD, not the numerical average from the data.

        Returns:
            :py:attr:`array_like`: MSD values
        """
        return self._diff.n

    @property
    def tmsd_std(self):
        """
        Returns MSD standard deviations values for the input trajectories.

        Returns:
            :py:attr:`array_like`: Standard deviation values for MSD
        """
        return self._diff.s

    @property
    def D_J(self):
        """
        Diffusion coefficient.

        Returns:
            :py:class:`uravu.distribution.Distribution`: Jump diffusion coefficient
        """
        return self._diff.jump_diffusion_coefficient


class ConductivityAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.ConductivityAnalyzer` class performs analysis of conductive relationships in materials.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically accurate values for mean squared charge displacement uncertainty and covariance.
    The time-dependence of the MSCD is then modelled in a generalised least squares fashion to obtain the jump diffusion coefficient and offset using Markov chain Monte Carlo maximum likelihood sampling.

    Args:
        trajectory (:py:attr:`str` or :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:class:`MDAnalysis.core.Universe` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or :py:attr:`list` of :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure`): The trajectory/ies to be analysed, supported trajectories are those from VASP (as file paths to be read, :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects, or a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects) or in the format of an :py:class:`MDAnalysis.core.Universe` (:py:attr:`dtype` should contain :py:code:`vasp` or :py:code:`mdanalysis` as appropriate). Additionally, for VASP trajectories, a list of identical starting points can be passed by using :py:code:`dtype='identicalvasp'` and consecutive VASP trajectories using :py:code:`dtype='consecutivevasp'`
        ionic_charge (:py:attr:`array_like` or :py:attr:`int`): The charge on the mobile ions, either an array with a value for each ion or a scalar if all values are the same
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this dictionary
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters
        dtype (:py:attr:`str`, optional): The file format of the :py:attr:`trajectory`, see documentation for :py:attr:`trajectory` above. Defaults to :py:attr:`'vasp'`
    """
    def __init__(self, trajectory, ionic_charge, parser_params, bootstrap_params=None, dtype='vasp'):
        if bootstrap_params is None:
            bootstrap_params = {}
        super().__init__(trajectory, parser_params, dtype)
        self._diff = diffusion.MSCDBootstrap(self.delta_t, self.disp_3d, ionic_charge, **bootstrap_params)

    def conductivity(self, temperature, conductivity_params=None):
        """
        Calculate the jump diffusion coefficicent using the bootstrap-GLS methodology.

        Args:
            temperature (:py:attr:`float`): Simulation temperature in Kelvin
            conductivity_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.MSCDBootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters
        """
        if conductivity_params is None:
            conductivity_params = {}
        self._diff.conductivity(temperature, self.volume, **conductivity_params)

    @property
    def dt(self):
        """
        Returns the timestep values that have been sampled.

        Returns:
            :py:attr:`array_like`: Timestep values
        """
        return self._diff.dt

    @property
    def mscd(self):
        """
        Returns MSCD for the input trajectories. Note that this is the bootstrap sampled value, not the numerical average from the data.

        Returns:
            :py:attr:`array_like`: MSD values
        """
        return self._diff.n

    @property
    def mscd_std(self):
        """
        Returns MSCD standard deviations values for the input trajectories.

        Returns:
            :py:attr:`array_like`: Standard deviation values for MSD
        """
        return self._diff.s

    @property
    def sigma(self):
        """
        Conductivity, in mS^{1}cm^{-1}.

        Returns:
            :py:class:`uravu.distribution.Distribution`: Conductivity
        """
        return self._diff.sigma


def _flatten_list(this_list):
    """
    Flatten nested lists.

    Args:
        this_list (:py:attr:`list`): List to be flattened

    Returns:
        :py:attr:`list`: Flattened list
    """
    return [item for sublist in this_list for item in sublist]
