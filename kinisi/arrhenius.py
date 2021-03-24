"""
The two classes herein enable the determination of the activation energy from a diffusion-Arrhenius or -Super-Arrhenius plot.
This includes the uncertainty on the activation energy from the MCMC sampling of the plot, with uncertainties on diffusion.
Furthermore, the classes are build on the Relationship subclass, therefore it is easy to determine the Bayesian evidence for each of the models with the given data, enabling the distinction between Arrhenius and Super-Arrhenius diffusion.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913

import sys
import numpy as np
from uravu.relationship import Relationship
from scipy.constants import R, N_A, eV
from scipy.stats import uniform

# Convert R to eV
R_no_mol = R / N_A
R_eV = R_no_mol / eV


class StandardArrhenius(Relationship):
    """
    Evaluate the data with a standard Arrhenius relationship.
    For attributes associated with the :py:class:`uravu.relationship.Relationship` class see that documentation.
    The :py:attr:`uravu.relationship.Relationship.variables` for this model is a :py:attr:`list` of length 2, where :py:attr:`kinisi.arrhenius.StandardArrhenius.variables[0]` is the activation energy (in eV) and :py:attr:`kinisi.arrhenius.StandardArrhenius.variables[1]` is the prefactor of the Arrhenius equation.

    Args:
        temperature (:py:attr:`array_like`): Temperature data.
        diffusion (:py:attr:`array_like`): Diffusion coefficient data.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters. Defaults to :py:attr:`None`.
        diffusion_error (:py:attr:`array_like`): Uncertainty in the diffusion coefficient data. Not necessary if :py:attr:`diffusion` is :py:attr:`list` of :py:class:`uravu.distribution.Distribution` objects.
        ci_points (:py:attr:`array_like`, optional): The two percentiles at which confidence intervals should be found for the variables. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
    """
    def __init__(self, temperature, diffusion, bounds, diffusion_error=None, ci_points=None):
        super().__init__(arrhenius, temperature, diffusion, bounds, ordinate_error=diffusion_error, ci_points=None)


def arrhenius(abscissa, activation_energy, prefactor):
    """
    Determine the diffusion coefficient for a given activation energy, and prefactor according to the Arrhenius equation.

    Args:
        abscissa (:py:attr:`array_like`): The abscissa data.
        activation_energy (:py:attr:`float`): The activation_energy value.
        prefactor (:py:attr:`float`): The prefactor value.

    Returns:
        :py:attr:`array_like`: The diffusion coefficient data.
    """
    return prefactor * np.exp(-1 * activation_energy / (R_eV * abscissa))


class SuperArrhenius(Relationship):
    """
    Evaluate the data with a super-Arrhenius relationship. For attributes associated with the :py:class:`uravu.relationship.Relationship` class see that documentation.
    This :py:attr:`uravu.relationship.Relationship.variables` for this model is a :py:attr:`list` of length 3, where :py:attr:`~uravu.relationship.Relationship.variables[0]` is the activation energy (in eV), :py:attr:`~uravu.relationship.Relationship.variables[1]` is the prefactor of the Arrhenius equation, and :py:attr:`~uravu.relationship.Relationship.variables[2]` is the temperature offset (in Kelvin).

    Args:
        temperature (:py:attr:`array_like`): Temperature data.
        diffusion (:py:attr:`array_like`): Diffusion coefficient data.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters. Defaults to :py:attr:`None`.
        diffusion_error (:py:attr:`array_like`): Uncertainty in the diffusion coefficient data. Not necessary if :py:attr:`diffusion` is :py:attr:`list` of :py:class:`uravu.distribution.Distribution` objects.
        ci_points (:py:attr:`array_like`, optional): The two percentiles at which confidence intervals should be found for the variables. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
    """
    def __init__(self, temperature, diffusion, bounds, diffusion_error=None, ci_points=None):
        super().__init__(super_arrhenius, temperature, diffusion, bounds, ordinate_error=diffusion_error, ci_points=None)


def super_arrhenius(abscissa, activation_energy, prefactor, t_zero):
    """
    Determine the rate constant for a given activation energy, prefactor, and t_zero according to the Vogel–Tammann–Fulcher equation.

    Args:
        abscissa (:py:attr:`array_like`): The abscissa data.
        activation_energy (:py:attr:`float`): The activation_energy value.
        prefactor (:py:attr:`float`): The prefactor value.
        t_zero (:py:attr:`float`): The T_0 value.

    Returns:
        :py:attr:`array_like`: The diffusion coefficient data.
    """
    return prefactor * np.exp(
        -1 * activation_energy / (R_eV * (abscissa - t_zero)))
