"""
The :py:class:kinisi.analyze.ConductivityAnalyzer` class for MSCD 
and conductivity analysis.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import Union, List
import numpy as np
import scipp as sc
from scipp.typing import VariableLikeType
from kinisi.diffusion import Diffusion
from kinisi.displacement import calculate_mscd
from kinisi.parser import Parser
from kinisi.analyzer import Analyzer


class ConductivityAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.ConductivityAnalyzer` class performs analysis of conductivity in materials. 
    This is achieved through the application of a Bayesian regression methodology to obtain optimal estimates 
    of the mean-squared charge displacement and covariance. 
    This is then sampled with a linear model to estimate the conductivity, via the jump diffusion coefficient. 
    
    :param trajectory: The parsed trajectory from some input file. This will be of type :py:class:`Parser`, but
        the specifics depend on the parser that is used.
    """

    def __init__(self, trajectory: Parser) -> None:
        super().__init__(trajectory)
        self.msd_da = None

    @classmethod
    def from_xdatcar(cls,
                     trajectory: Union['pymatgen.io.vasp.outputs.Xdatcar', List['pymatgen.io.vasp.outputs.Xdatcar']],
                     specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
                     time_step: VariableLikeType,
                     step_skip: VariableLikeType,
                     ionic_charge: VariableLikeType,
                     dtype: Union[str, None] = None,
                     dt: VariableLikeType = None,
                     dimension: str = 'xyz',
                     distance_unit: sc.Unit = sc.units.angstrom,
                     progress: bool = True) -> 'ConductivityAnalyzer':
        """
        Constructs the necessary :py:mod:`kinisi` objects for analysis from a single or a list of
        :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects.

        :param trajectory: The :py:class:`pymatgen.io.vasp.outputs.Xdatcar` or list of these that should be parsed. 
        :param specie: Specie to calculate diffusivity for as a String, e.g. :py:attr:`'Li'`.
        :param time_step: The input simulation time step, i.e., the time step for the molecular dynamics integrator. Note, 
            that this must be given as a :py:mod:`scipp`-type scalar. The unit used for the time_step, will be the unit 
            that is use for the time interval values.
        :param step_skip: Sampling freqency of the simulation trajectory, i.e., how many time steps exist between the
            output of the positions in the trajectory. Similar to the :py:attr:`time_step`, this parameter must be
            a :py:mod:`scipp` scalar. The units for this scalar should be dimensionless.
        :param ionic_charge: The ionic charge of the species of interest. This should be either a :py:mod:`scipp`
            scalar if all of the ions have the same charge or an array of the charge for each indiviudal ion.
        :param dtype: If :py:attr:`trajectory` is a :py:class:`pymatgen.io.vasp.outputs.Xdatcar` object, this should
            be :py:attr:`None`. However, if a list of :py:class:`pymatgen.io.vasp.outputs.Xdatcar` objects is passed,
            then it is necessary to identify if these constitute a series of :py:attr:`consecutive` trajectories or
            a series of :py:attr:`identical` starting points with different random seeds, in which case the `dtype`
            should be either :py:attr:`consecutive` or :py:attr:`identical`.:
        :param dt: Time intervals to calculate the displacements over. Optional, defaults to a :py:mod:`scipp` array
            ranging from the smallest interval (i.e., time_step * step_skip) to the full simulation length, with 
            a step size the same as the smallest interval.
        :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
            the axes of interest. Optional, defaults to `'xyz'`.
        :param distance_unit: The unit of distance in the simulation input. This should be a :py:mod:`scipp` unit and
            defaults to :py:attr:`sc.units.angstrom`.
        :param progress: Print progress bars to screen. Optional, defaults to :py:attr:`True`.
        
        :returns: The :py:class:`ConductivityAnalyzer` object with the mean-squared charge displacement calculated.
        """
        p = super()._from_xdatcar(trajectory, specie, time_step, step_skip, dtype, dt, dimension, distance_unit,
                                  progress)
        p.msd_da = calculate_mscd(p.trajectory, ionic_charge, progress)
        return p

    def conductivity(self,
                     start_dt: VariableLikeType,
                     temperature: VariableLikeType,
                     volume: VariableLikeType,
                     diffusion_params: Union[dict, None] = None) -> None:
        """
        Calculation of the conductivity.
        Keyword arguments will be passed of the :py:func:`bayesian_regression` method. 

        :param start_dt: The time at which the diffusion regime begins.
        :param temperature: The temperature of the system.
        :param volume: The volume of the system.
        :param kwargs: Additional keyword arguments to pass to :py:func:`bayesian_regression`.
        """
        if diffusion_params is None:
            diffusion_params = {}
        self.diff = Diffusion(msd=self.msd_da, n_atoms=self.n_atoms)
        self.diff._conductivity(start_dt=start_dt, temperature=temperature, volume=volume, **diffusion_params)

    @property
    def distributions(self) -> np.array:
        """
        :return: A distribution of samples for the linear relationship that can be used for easy
        plotting of credible intervals.
        """
        if self.diff.intercept is not None:
            return self.diff.gradient.values * self.msd_da.coords[
                'time interval'].values[:, np.newaxis] + self.diff.intercept.values
        else:
            return self.diff.gradient.values * self.msd_da.coords['time interval'].values[:, np.newaxis]

    @property
    def sigma(self) -> VariableLikeType:
        """
        :return: The conductivity.
        """
        return self.diff.sigma

    @property
    def mscd(self) -> VariableLikeType:
        """
        :return: The mean-squared charge displacement.
        """
        return self.msd_da.data

    @property
    def flatchain(self) -> sc.DataGroup:
        """
        :returns: The flatchain of the MCMC samples.
        """
        flatchain = {'sigma': self.sigma}
        if self.intercept is not None:
            flatchain['intercept'] = self.intercept
        return sc.DataGroup(**flatchain)
