"""
This module contains the API classes for :py:mod:`kinisi`. 
It is anticipated that this is where the majority of interaction with the package will occur. 
This module includes the :py:class:`~kinisi.analyze.DiffAnalyzer` class for diffusion analysis, which is compatible with both VASP Xdatcar output files and any MD trajectory that the :py:mod:`MDAnalysis` package can handle. 
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import MDAnalysis as mda
from pymatgen.io.vasp import Xdatcar
from uncertainties import unumpy as unp
from kinisi import diffusion
from kinisi.parser import MDAnalysisParser, PymatgenParser

class DiffAnalyzer:
    """
    The :py:class:`kinisi.analyze.DiffAnalyzer` class performs analysis of diffusion relationships in materials. 
    This is achieved through the application of a block bootstrapping methodology to obtain the most statistically accurate values for mean squared displacement and the associated uncertainity. 
    The time-scale dependence of the MSD is then modeled with a straight line Einstein relationship, and Markov chain Monte Carlo is used to quantify inverse uncertainties for this model. 

    Attributes:
        delta_t (:py:attr:`array_like`):  Timestep values. 
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): Each element in the :py:attr:`list` has the axes [atom, displacement observation, dimension] and there is one element for each delta_t value. *Note: it is necessary to use a :py:attr:`list` of :py:attr:`array_like` as the number of observations is not necessary the same at each timestep point*.
        indices (:py:attr:`array_like`): Indices for the atoms in the trajectory used in the calculation of the diffusion.
        msd (:py:attr:`array_like`): The block bootstrap determined mean squared displacement values.
        msd_err (:py:attr:`array_like`): A single standard deviation on each distribution underlying the mean squared displacement values.
        msd_lb (:py:attr:`array_like`): The 2.5 % confidence interval value on each distribution underlying the mean squared displacement values.
        msd_ub (:py:attr:`array_like`): The 97.5 % confidence interval value on each distribution underlying the mean squared displacement values.
        relationship (:py:class:`kinisi.diffusion.Diffusion`): The :py:class:`~kinisi.diffusion.Diffusion` class object that describes the diffusion Einstein relationship.
        D (:py:class:`uravu.distribution.Distribution`): The gradient of the Einstein relationship divided by 6 (twice the number of dimensions).
        D_offset (:py:class:`uravu.distribution.Distribution`): The offset from the abscissa of the Einstein relationship.

    Args:
        file (:py:attr:`str` or :py:attr:`list` of :py:attr:`str`): The file path(s) that should be read by either the :py:class:`pymatgen.io.vasp.Xdatcar` or :py:class:`MDAnalysis.core.universe.Universe` classes. 
        params (:py:attr:`dict`): The parameters for the :py:mod:`kinisi.parser` object, which is either :py:class:`kinisi.parser.PymatgenParser` or :py:class:`kinisi.parser.MDAnalysisParser` depending on the input file format. See the appropriate documention for more guidence on this object.  
        format (:py:attr:`str`, optional): The file format, for the :py:class:`kinisi.parser.PymatgenParser` this should be :py:attr:`'Xdatcar'` and for :py:class:`kinisi.parser.MDAnalysisParser` this should be the appropriate format to be passed to the :py:class:`MDAnalysis.core.universe.Universe`. Defaults to :py:attr:`'Xdatcar'`.
        bounds (:py:attr:`tuple`, optional): Minimum and maximum values for the gradient and intercept of the diffusion relationship. Defaults to :py:attr:`((0, 100), (-10, 10))`. 
    """
    def __init__(self, file, params, format='Xdatcar', bounds=((0, 100), (-10, 10))):  # pragma: no cover
        if format is 'Xdatcar':
            xd = Xdatcar(file)
            u = PymatgenParser(xd.structures, **params)
        else:
            universe = mda.Universe(*file, format=format)
            u = MDAnalysisParser(universe, **params)

        self.delta_t = u.delta_t
        self.disp_3d = u.disp_3d
        self.indices = u.indices

        diff_data = diffusion.msd_bootstrap(self.delta_t, self.disp_3d)

        self.delta_t = diff_data[0]
        self.MSD = diff_data[3]

        self.relationship = diffusion.Diffusion(self.delta_t, self.MSD, bounds)

        self.msd = self.relationship.y.n
        self.msd_err = self.relationship.y.s

        self.relationship.max_likelihood()
        self.relationship.sample()

        self.D = self.relationship.diffusion_coefficient
        self.D_offset = self.relationship.variables[1]
