"""
The :py:class:`kinisi.analyze.ConductivityAnalyzer` class will allow the conductivity of a material in a
simulation to be found, without assuming a Haven ration of 1.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import Union, List
import numpy as np
from kinisi import diffusion
from .analyzer import Analyzer


class ConductivityAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.ConductivityAnalyzer` class performs analysis of conductive relationships in
    materials.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically
    accurate values for mean squared charge displacement uncertainty and covariance.
    The time-dependence of the MSCD is then modelled in a generalised least squares fashion to obtain the jump
    diffusion coefficient and offset using Markov chain Monte Carlo maximum likelihood sampling.

    :param delta_t: An array of the timestep values.
    :param disp_3d: A list of arrays, where each array has the axes [atom, displacement observation, dimension].
        There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as
        the number of observations is not necessary the same at each data point.
    :param volume: The volume of the simulation cell.
    :param bootstrap_params: The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See
        the appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.
    :param ionic_charge: The charge on the mobile ions, either an array with a value for each ion or a scalar
        if all values are the same. Optional, default is :py:attr:`1`.
    """

    def __init__(self, delta_t: np.ndarray, disp_3d: List[np.ndarray], n_o: np.ndarray, volume: float):
        super().__init__(delta_t, disp_3d, n_o, volume)
        self._diff = None

    def to_dict(self) -> dict:
        """
        :return: Dictionary description of :py:class:`ConductivityAnalyzer`.
        """
        my_dict = super().to_dict()
        my_dict['diff'] = self._diff.to_dict()
        return my_dict

    @classmethod
    def from_dict(cls, my_dict: dict) -> 'ConductivityAnalyzer':
        """
        Generate a :py:class:`ConductivityAnalyzer` object from a dictionary.

        :param my_dict: The input dictionary.

        :return: New :py:class`ConductivityAnalyzer` object.
        """
        cond_anal = cls(my_dict['delta_t'], my_dict['disp_3d'], my_dict['n_o'], my_dict['volume'])
        cond_anal._diff = diffusion.Bootstrap.from_dict(my_dict['diff'])
        return cond_anal

    @classmethod
    def from_pymatgen(cls,
                      trajectory: List[Union['pymatgen.core.structure.Structure',
                                             List['pymatgen.core.structure.Structure']]],
                      parser_params: dict,
                      dtype: str = None,
                      bootstrap_params: dict = None,
                      ionic_charge: Union[np.ndarray, int] = 1):
        """
        Create a :py:class:`ConductivityAnalyzer` object from a list or nested list of
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
        :param ionic_charge: The charge on the mobile ions, either an array with a value for each ion or a scalar
            if all values are the same. Optional, default is :py:attr:`1`.

        :return: Relevant :py:class:`ConductivityAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        cond_anal = super()._from_pymatgen(trajectory, parser_params, dtype=dtype)
        cond_anal._diff = diffusion.MSCDBootstrap(cond_anal._delta_t, cond_anal._disp_3d, ionic_charge, cond_anal._n_o,
                                                  **bootstrap_params)
        return cond_anal

    @classmethod
    def from_Xdatcar(cls,
                     trajectory: Union['pymatgen.io.vasp.outputs.Xdatcar', List['pymatgen.io.vasp.outputs.Xdatcar']],
                     parser_params: dict,
                     dtype: str = None,
                     bootstrap_params: dict = None,
                     ionic_charge: Union[np.ndarray, int] = 1):
        """
        Create a :py:class:`ConductivityAnalyzer` object from a single or a list of
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
        :param ionic_charge: The charge on the mobile ions, either an array with a value for each ion or a scalar
            if all values are the same. Optional, default is :py:attr:`1`.

        :return: Relevant :py:class:`ConductivityAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        cond_anal = super()._from_Xdatcar(trajectory, parser_params, dtype=dtype)
        cond_anal._diff = diffusion.MSCDBootstrap(cond_anal._delta_t, cond_anal._disp_3d, ionic_charge, cond_anal._n_o,
                                                  **bootstrap_params)
        return cond_anal

    @classmethod
    def from_file(cls,
                  trajectory: Union[str, List[str]],
                  parser_params: dict,
                  dtype: str = None,
                  bootstrap_params: dict = None,
                  ionic_charge: Union[np.ndarray, int] = 1):
        """
        Create a :py:class:`ConductivityAnalyzer` object from a single or a list of Xdatcar file(s).

        :param trajectory: The file or list of Xdatcar files to be analysed.
        :param parser_params: The parameters for the :py:class:`kinisi.parser.PymatgenParser` object. See the
            appropriate documentation for more guidance on this dictionary.
        :param dtype: If :py:attr:`trajectory` is a single file, this should be :py:attr:`None`. However, if a
            list of files is passed, then it is necessary to identify if these constitute a series of
            :py:attr:`consecutive` trajectories or a series of :py:attr:`identical` starting points with different
            random seeds, in which case the `dtype` should be either :py:attr:`consecutive` or :py:attr:`identical`.
        :param bootstrap_params: The parameters for the :py:class:`kinisi.diffusion.MSDBootstrap` object. See the
            appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.
        :param ionic_charge: The charge on the mobile ions, either an array with a value for each ion or a scalar
            if all values are the same. Optional, default is :py:attr:`1`.

        :return: Relevant :py:class:`ConductivityAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        cond_anal = super()._from_file(trajectory, parser_params, dtype=dtype)
        cond_anal._diff = diffusion.MSCDBootstrap(cond_anal._delta_t, cond_anal._disp_3d, ionic_charge, cond_anal._n_o,
                                                  **bootstrap_params)
        return cond_anal

    @classmethod
    def from_universe(cls,
                      trajectory: 'MDAnalysis.core.universe.Universe',
                      parser_params: dict,
                      dtype: str = None,
                      bootstrap_params: dict = None,
                      ionic_charge: Union[np.ndarray, int] = 1):
        """
        Create an :py:class:`ConductivityAnalyzer` object from an :py:class:`MDAnalysis.core.universe.Universe` object.

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
        :param ionic_charge: The charge on the mobile ions, either an array with a value for each ion or a scalar
            if all values are the same. Optional, default is :py:attr:`1`.

        :return: Relevant :py:class:`ConductivityAnalyzer` object.
        """
        if bootstrap_params is None:
            bootstrap_params = {}
        cond_anal = super()._from_universe(trajectory, parser_params, dtype=dtype)
        cond_anal._diff = diffusion.MSCDBootstrap(cond_anal._delta_t, cond_anal._disp_3d, ionic_charge, cond_anal._n_o,
                                                  **bootstrap_params)
        return cond_anal

    def conductivity(self, temperature: float, conductivity_params: Union[dict, None] = None):
        """
        Calculate the jump diffusion coefficicent using the bootstrap-GLS methodology.

        :param temperature: Simulation temperature in Kelvin
        :param conductivity_params: The parameters for the :py:class:`kinisi.diffusion.MSCDBootstrap` object.
            See the appropriate documentation for more guidance on this. Optional, default is the default
            bootstrap parameters
        """
        if conductivity_params is None:
            conductivity_params = {}
        self._diff.conductivity(temperature, self._volume, **conductivity_params)

    @property
    def mscd(self) -> np.ndarray:
        """
        :return: MSCD for the input trajectories. Note that this is the bootstrap sampled value, not the numerical
            average from the data.
        """
        return self._diff.n

    @property
    def mscd_std(self) -> np.ndarray:
        """
        :return: MSCD standard deviation values for the input trajectories (a single standard deviation).
        """
        return self._diff.s

    @property
    def sigma(self) -> 'uravu.distribution.Distribution':
        """
        :returns: Conductivity, in mS^{1}cm^{-1}.
        """
        return self._diff.sigma

    @property
    def flatchain(self) -> np.ndarray:
        """
        :return: sampling flatchain
        """
        return np.array([self.sigma.samples, self.intercept.samples]).T
