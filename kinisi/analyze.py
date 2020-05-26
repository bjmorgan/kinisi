"""
This module contains the API classes for :py:mod:`kinisi`. 
It is anticipated that this is where the majority of interaction with the package will occur. 
This module includes the :py:class:`~kinisi.analyze.DiffAnalyzer` class for diffusion analysis, which is compatible with both VASP Xdatcar output files and any MD trajectory that the :py:mod:`MDAnalysis` package can handle. 
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
import MDAnalysis as mda
from pymatgen.io.vasp import Xdatcar
from pymatgen.analysis.diffusion_analyzer import get_conversion_factor
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
        relationship (:py:class:`kinisi.diffusion.Diffusion`): The :py:class:`~kinisi.diffusion.Diffusion` class object that describes the diffusion Einstein relationship.
        msd_observed (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations.
        msd_sampled (:py:attr:`array_like`): The population mean-squared displacements, found from the bootstrap resampling of the observations.
        msd_sampled_err (:py:attr:`array_like`): The two-dimensional uncertainties, at the given confidence interval, found from the bootstrap resampling of the observations.

    Args:
        trajectory (:py:attr:`str` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes, or a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects ordered in sequence of run. 
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this object.
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.Bootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters.
        dtype (:py:attr:`str`, optional): The file format, for the :py:class:`kinisi.parser.PymatgenParser` this should be :py:attr:`'Xdatcar'` and for :py:class:`kinisi.parser.MDAnalysisParser` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`. Defaults to :py:attr:`'Xdatcar'`.
        charge (:py:attr:`bool`, optional): Calculate the charge displacement. Default is :py:attr:`False`.
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, dtype='Xdatcar', charge=False):  # pragma: no cover
        if bootstrap_params is None:
            bootstrap_params = {}
        if dtype is 'Xdatcar':
            if isinstance(trajectory, list):
                trajectory_list = (Xdatcar(f) for f in trajectory)
                structures = _flatten_list([x.structures for x in trajectory_list])
            else:
                xd = Xdatcar(trajectory)
                structures = xd.structures
            u = PymatgenParser(structures, **parser_params)
            self.first_structure = structures[0]
        elif dtype is 'structures':
            u = PymatgenParser(trajectory, **parser_params)
            self.first_structure = structures[0]
        else:
            universe = mda.Universe(*trajectory, format=dtype)
            u = MDAnalysisParser(universe, **parser_params)

        dt = u.delta_t
        disp_3d = u.disp_3d

        if charge:
            diff_data = diffusion.MSCDBootstrap(dt, disp_3d, **bootstrap_params)
        else:
            diff_data = diffusion.MSDBootstrap(dt, disp_3d, **bootstrap_params)

        self.dt = diff_data.dt
        self.msd_sampled = diff_data.msd_sampled
        self.msd_sampled_err = diff_data.msd_sampled_err
        self.msd_observed = diff_data.msd_observed
        self.distributions = diff_data.distributions

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
            :py:attr:`array_like`: A lower and upper uncertainty, at a 95 % confidence interval, of the mean squared displacement values..
        """
        return self.msd_sampled_err

    @property
    def con_int(self):
        """
        Returns MSD confidence inteval, at 95 %, for the input trajectories.

        Returns:
            :py:attr:`array_like`: A lower and upper 95 % confidence interval of the mean squared displacement values..
        """ 
        return np.array([self.msd - self.msd_err[0], self.msd + self.msd_err[1]])


class DiffAnalyzer(MSDAnalyzer):
    """
    The :py:class:`kinisi.analyze.DiffAnalyzer` class performs analysis of diffusion relationships in materials. 
    This is achieved through the application of a block bootstrapping methodology to obtain the most statistically accurate values for mean squared displacement and the associated uncertainty. 
    The time-scale dependence of the MSD is then modeled with a straight line Einstein relationship, and Markov chain Monte Carlo is used to quantify inverse uncertainties for this model. 

    Args:
        trajectory (:py:attr:`str` or :py:attr:`list` of :py:attr:`str` or :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes, or a :py:attr:`list` of :py:class:`pymatgen.core.structure.Structure` objects ordered in sequence of run. 
        parser_params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidance on this object.
        bootstrap_params (:py:attr:`dict`, optional): The parameters for the :py:class:`kinisi.diffusion.Bootstrap` object. See the appropriate documentation for more guidance on this. Default is the default bootstrap parameters.
        dtype (:py:attr:`str`, optional): The file format, for the :py:class:`kinisi.parser.PymatgenParser` this should be :py:attr:`'Xdatcar'` and for :py:class:`kinisi.parser.MDAnalysisParser` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`. Defaults to :py:attr:`'Xdatcar'`.
        bounds (:py:attr:`tuple`, optional): Minimum and maximum values for the gradient and intercept of the diffusion relationship. Defaults to :py:attr:`((0, 100), (-10, 10))`. 
        sampling_method (:py:attr:`str`, optional): The method used to sample the posterior distributions. Can be either :py:attr:`'mcmc'` or :py:attr:`'nested_sampling'`. Default is :py:attr:`'mcmc'`.
        sampling_kwargs (:py:attr:`dict`, optional): Keyword arguments to be passed to the sampling method. See :py:class:`uravu.relationship.Relationship` for options.
        charge (:py:attr:`bool`, optional): Calculate the charge mean-squared displacment. Default is :py:attr:`False`.
    """
    def __init__(self, trajectory, parser_params, bootstrap_params=None, dtype='Xdatcar', bounds=((0, 100), (-10, 10)), sampling_method='mcmc', sampling_kwargs={}, charge=False):  # pragma: no cover
        self.pymatgen = False
        if dtype == 'Xdatcar' or dtype == 'structures':
            self.pymatgen = True
            self.specie = parser_params['specie']
        super().__init__(trajectory, parser_params, bootstrap_params, dtype, charge) 
        self.relationship = diffusion.Diffusion(self.dt, self.distributions, bounds=bounds) 
   
        self.relationship.bounds = bounds
        self.relationship.max_likelihood('diff_evo')
        if sampling_method == 'mcmc':
            self.relationship.mcmc(**sampling_kwargs)
        elif sampling_method == 'nested_sampling':
            self.relationship.nested_sampling(**sampling_kwargs)
        else:
            raise ValueError("The only available sampling methods are 'mcmc' or 'nested_sampling', please select one of these.")

    @property
    def D(self):
        """
        Diffusion coefficient. 

        Returns:
            :py:class:`uravu.distribution.Distribution`: Diffusion coefficient.
        """
        return self.relationship.diffusion_coefficient

    @property
    def D_offset(self):
        """
        Offset from abscissa. 

        Returns:
            :py:class:`uravu.distribution.Distribution`: Abscissa offset.
        """
        return self.relationship.variables[1]

    def sigma(self, temperature):
        """
        Conductivity.

        Args:
            :py:attr:`float`: Simulation temperature.

        Returns:
            :py:class:`uravu.distribution.Distribution`: Conductivity.
        """
        if self.pymatgen:
            conv_factor = get_conversion_factor(self.first_structure, self.specie, temperature)
            return Distribution(self.relationship.diffusion_coefficient.samples * conv_factor)
        else:
            raise ValueError("This is currently only supported for Pymatgen files, please convert from D manually.")


def _flatten_list(this_list):
    """
    Flatten nested lists.

    Args:
        this_list (:py:attr:`list`): List to be flattened. 

    Returns:
        :py:attr:`list`: Flattened list.
    """
    return [item for sublist in this_list for item in sublist]