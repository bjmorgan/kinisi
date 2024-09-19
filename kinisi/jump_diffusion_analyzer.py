"""
The :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` class allows the study of jump diffusion
and the collective motion of particles.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import Union, List
import numpy as np
import scipp as sc
from kinisi.displacement import calculate_mstd, n_atoms
from kinisi.diffusion import Diffusion
from kinisi.parser import Parser, PymatgenParser
from kinisi.analyzer import Analyzer


class JumpDiffusionAnalyzer(Analyzer):
    """
    The :py:class:`kinisi.analyze.JumpDiffusionAnalyzer` class performs analysis of collective diffusion
    relationships in materials.
    This is achieved through the application of a bootstrapping methodology to obtain the most statistically
    accurate values for total mean squared displacement uncertainty and covariance.
    The time-dependence of the MSTD is then modelled in a generalised least squares fashion to obtain the jump
    diffusion coefficient and offset using Markov chain Monte Carlo maximum likelihood sampling.

    :param delta_t: An array of the time interval values.
    :param disp_3d: A list of arrays, where each array has the axes [atom, displacement observation, dimension].
        There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as
        the number of observations is not necessary the same at each data point.
    :param volume: The volume of the simulation cell.
    :param uncertainty_params: The parameters for the :py:class:`kinisi.diffusion.DiffBootstrap` object. See
        the appropriate documentation for more guidance on this. Optional, default is the default bootstrap parameters.
    """

    def __init__(self, trajectory: Parser) -> None:
        super().__init__(trajectory)
        self.mstd = None

    @classmethod
    def from_Xdatcar(cls,
                     trajectory: Union['pymatgen.io.vasp.outputs.Xdatcar',
                                       List['pymatgen.io.vasp.outputs.Xdatcar']],
                     specie: Union['pymatgen.core.periodic_table.Element', 'pymatgen.core.periodic_table.Specie'],
                     time_step: sc.Variable,
                     step_skip: sc.Variable,
                     dtype: Union[str, None] = None,
                     dt: sc.Variable = None,
                     dimension: str = 'xyz',
                     distance_unit: sc.Unit = sc.units.angstrom,
                     progress: bool = True) -> 'DiffusionAnalyzer':
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
        
        :returns: The :py:class:`DiffusionAnalyzer` object with the mean-squared displacement calculated.
        """
        p = super()._from_Xdatcar(trajectory, specie, time_step, step_skip, dtype, dt, dimension, distance_unit, progress)
        p.mstd = calculate_mstd(p.trajectory, progress)
        p.n_atoms = n_atoms(p.trajectory)
        return p


    def jump_diffusion(self, start_dt: sc.Variable, diffusion_params: Union[dict, None] = None) -> None:
        """
        Calculate the jump diffusion coefficient using the mean-squared total displacement data.
        
        :param start_dt: The time at which the diffusion regime begins.
        :param diffusion_params: The keyword arguements for the diffusion calculation
            (see :py:func:`diffusion.bayesian_regression`). Optional, defaults to :py:attr:`None`.
        """
        if diffusion_params is None:
            diffusion_params = {}
        self.diff = Diffusion(msd = self.mstd, n_atoms = self.n_atoms)
        self.diff.jump_diffusion(start_dt, **diffusion_params)
