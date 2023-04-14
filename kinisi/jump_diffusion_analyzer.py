"""
The :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` class allows the study of jump diffusion
and the collective motion of particles.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import Union, List
import numpy as np
from kinisi import diffusion
from .analyzer import Analyzer


class JumpDiffusionAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` class performs analysis of collective diffusion
    relationships in materials.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically
    accurate values for total mean squared displacement uncertainty and covariance.
    The time-dependence of the MSTD is then modelled in a generalised least squares fashion to obtain the jump
    diffusion coefficient and offset using Markov chain Monte Carlo maximum likelihood sampling.

    :param delta_t: An array of the timestep values.
    :param disp_3d: A list of arrays, where each array has the axes [atom, displacement observation, dimension].
        There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as
        the number of observations is not necessary the same at each data point.
    :param volume: The volume of the simulation cell.
    :param bootstrap_params: The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See
        the appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.
    """

    def __init__(self, delta_t: np.ndarray, disp_3d: List[np.ndarray], n_o: np.ndarray, volume: float):
        super().__init__(delta_t, disp_3d, n_o, volume)
        self._diff = None

    def to_dict(self) -> dict:
        """
        :return: Dictionary description of :py:class:`JumpDiffusionAnalyzer`.
        """
        my_dict = super().to_dict()
        my_dict['diff'] = self._diff.to_dict()
        return my_dict

    @classmethod
    def from_dict(cls, my_dict) -> 'JumpDiffusionAnalyzer':
        """
        Generate a :py:class:`DiffusionAnalyzer` object from a dictionary.

        :param my_dict: The input dictionary.

        :return: New :py:class:`DiffusionAnalyzer` object.
        """
        jdiff_anal = cls(my_dict['delta_t'], my_dict['disp_3d'], my_dict['n_o'], my_dict['volume'])
        jdiff_anal._diff = diffusion.Bootstrap.from_dict(my_dict['diff'])
        return jdiff_anal

    @classmethod
    def from_pymatgen(cls,
                      trajectory: List[Union['pymatgen.core.structure.Structure',
                                             List['pymatgen.core.structure.Structure']]],
                      parser_params: dict,
                      dtype: str = None,
                      bootstrap_params: dict = None):
        """
        Create a :py:class:`JumpDiffusionAnalyzer` object from a list or nested list of
        :py:class:`pymatgen.core.structure.Structure` objects.

        :param trajectory: The list or nested list of structures to be analysed.
        :param parser_params: The parameters for the :py:class:`kinisi.parser.PymatgenParser` object. See the
            appropriate documentation for more guidance on this dictionary.
        :param dtype: If :py:attr:`trajectory` is a list of :py:class:`pymatgen.core.structure.Structure` objects,
            this should be :py:attr:`None`. However, if a list of lists is passed, then it is necessary to identify if
            these constitute a series of :py:attr:`consecutive` trajectories or a series of :py:attr:`identical`
            starting points with different random seeds, in which case the `dtype` should be either
            :py:attr:`consecutive` or :py:attr:`identical`.
        :param bootstrap_params: The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the
            appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.

        :return: Relevant :py:class:`JumpDiffusionAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        jdiff_anal = super()._from_pymatgen(trajectory, parser_params, dtype=dtype)
        jdiff_anal._diff = diffusion.MSTDBootstrap(jdiff_anal._delta_t, jdiff_anal._disp_3d, jdiff_anal._n_o,
                                                   **bootstrap_params)
        return jdiff_anal

    @classmethod
    def from_Xdatcar(cls,
                     trajectory: Union['pymatgen.io.vasp.outputs.Xdatcar', List['pymatgen.io.vasp.outputs.Xdatcar']],
                     parser_params: dict,
                     dtype: str = None,
                     bootstrap_params: dict = None):
        """
        Create a :py:class:`JumpDiffusionAnalyzer` object from a single or a list of
        :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects.

        :param trajectory: The Xdatcar or list of Xdatcar objects to be analysed.
        :param parser_params: The parameters for the :py:class:`kinisi.parser.PymatgenParser` object. See the
            appropriate documentation for more guidance on this dictionary.
        :param dtype: If :py:attr:`trajectory` is a :py:class:`pymatgen.io.vasp.outputs.Xdatcar` object, this should
            be :py:attr:`None`. However, if a list of :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects is passed,
            then it is necessary to identify if these constitute a series of :py:attr:`consecutive` trajectories or a
            series of :py:attr:`identical` starting points with different random seeds, in which case the `dtype`
            should be either :py:attr:`consecutive` or :py:attr:`identical`.
        :param bootstrap_params: The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the
            appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.

        :return: Relevant :py:class:`JumpDiffusionAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        jdiff_anal = super()._from_Xdatcar(trajectory, parser_params, dtype=dtype)
        jdiff_anal._diff = diffusion.MSTDBootstrap(jdiff_anal._delta_t, jdiff_anal._disp_3d, jdiff_anal._n_o,
                                                   **bootstrap_params)
        return jdiff_anal

    @classmethod
    def from_file(cls,
                  trajectory: Union[str, List[str]],
                  parser_params: dict,
                  dtype: str = None,
                  bootstrap_params: dict = None):
        """
        Create a :py:class:`JumpDiffusionAnalyzer` object from a single or a list of Xdatcar file(s).

        :param trajectory: The file or list of Xdatcar files to be analysed.
        :param parser_params: The parameters for the :py:class:`kinisi.parser.PymatgenParser` object. See the
            appropriate documentation for more guidance on this dictionary.
        :param dtype: If :py:attr:`trajectory` is a single file, this should be :py:attr:`None`. However, if a
            list of files is passed, then it is necessary to identify if these constitute a series of
            :py:attr:`consecutive` trajectories or a series of :py:attr:`identical` starting points with different
            random seeds, in which case the `dtype` should be either :py:attr:`consecutive` or :py:attr:`identical`.
        :param bootstrap_params: The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the
            appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.

        :return: Relevant :py:class:`JumpDiffusionAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        jdiff_anal = super()._from_file(trajectory, parser_params, dtype=dtype)
        jdiff_anal._diff = diffusion.MSTDBootstrap(jdiff_anal._delta_t, jdiff_anal._disp_3d, jdiff_anal._n_o,
                                                   **bootstrap_params)
        return jdiff_anal

    @classmethod
    def from_universe(cls,
                      trajectory: 'MDAnalysis.core.universe.Universe',
                      parser_params: dict,
                      dtype: str = None,
                      bootstrap_params: dict = None):
        """
        Create an :py:class:`JumpDiffusionAnalyzer` object from an :py:class:`MDAnalysis.core.universe.Universe` object.

        :param trajectory: The universe to be analysed.
        :param parser_params: The parameters for the :py:class:`kinisi.parser.MDAnalysisParser` object.
            See the appropriate documention for more guidance on this dictionary.
        :param dtype: If :py:attr:`trajectory` is a single file, this should be :py:attr:`None`. However, if a
            list of files is passed, then it is necessary to identify that these constitute a series of
            :py:attr:`identical` starting points with different random seeds, in which case the `dtype` should
            be :py:attr:`identical`. For a series of consecutive trajectories, please construct the relevant
            object using :py:mod:`MDAnalysis`.
        :param bootstrap_params: The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the
            appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.

        :return: Relevant :py:class:`JumpDiffusionAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        jdiff_anal = super()._from_universe(trajectory, parser_params, dtype=dtype)
        jdiff_anal._diff = diffusion.MSTDBootstrap(jdiff_anal._delta_t, jdiff_anal._disp_3d, jdiff_anal._n_o,
                                                   **bootstrap_params)
        return jdiff_anal

    def jump_diffusion(self, jump_diffusion_params: Union[dict, None] = None):
        """
        Calculate the jump diffusion coefficicent using the bootstrap-GLS methodology.

        :param ump_diffusion_params: The parameters for the :py:class:`kinisi.diffusion.MSTDBootstrap`
            object. See the appropriate documentation for more guidance on this. Optional, default is the
            default bootstrap parameters.
        """
        if jump_diffusion_params is None:
            jump_diffusion_params = {}
        self._diff.jump_diffusion(**jump_diffusion_params)

    @property
    def mstd(self) -> np.ndarray:
        """
        :return: MSTD for the input trajectories. Note that this is the bootstrap sampled MSD, not the numerical
            average from the data.
        """
        return self._diff.n

    @property
    def mstd_std(self) -> np.ndarray:
        """
        :return: MSD standard deviation values for the input trajectories (a single standard deviation).
        """
        return self._diff.s

    @property
    def D_J(self) -> 'uravu.distribution.Distribution':
        """
        :return: Jump diffusion coefficient
        """
        return self._diff.D_J

    @property
    def flatchain(self) -> np.ndarray:
        """
        :return: sampling flatchain
        """
        return np.array([self.D_J.samples, self.intercept.samples]).T
