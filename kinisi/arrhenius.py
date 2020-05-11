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
from uravu.distribution import Distribution
from uravu.relationship import Relationship
from scipy.constants import R, N_A, eV
from scipy.stats import uniform
from uravu import UREG

# Convert R to eV
R_no_mol = R / N_A
R_eV = R_no_mol / eV


class StandardArrhenius(Relationship):
    """
    Evaluate the data with a standard Arrhenius relationship. For attributes associated with the :py:class:`uravu.relationship.Relationship` class see that documentation.
    This :py:attr:`uravu.relationship.Relationship.variables` for this model is a :py:attr:`list` of length 2, where :py:attr:`~uravu.relationship.Relationship.variables[0]` is the activation energy  (in eV) and :py:attr:`~uravu.relationship.Relationship.variables[1]` is the prefactor of the Arrhenius equation. 

    Args:
        temperature (:py:attr:`array_like`): Temperature data.
        diffusion (:py:attr:`array_like`): Diffusion coefficient data.
        diffusion_error (:py:attr:`array_like`): Uncertainty in the diffusion coefficient data.
        temperature_unit (:py:class:`pint.unit.Unit`, optional): The unit for the temperature. Default is :py:attr:`kelvin`..
        diffusion_unit (:py:class:`pint.unit.Unit`, optional): The unit for the diffusion coefficient. Default is :py:attr:`centimetre**2 / second`.
        temperature_names (:py:attr:`str`, optional): The label for the temperature. Default is :py:attr:`$T$`.
        diffusion_names (:py:attr:`str`, optional): The label for the diffusion coefficient. Default is :py:attr:`$D$`.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters. Defaults to :py:attr:`None`.
        unaccounted_uncertainty (:py:attr:`bool`, optional): Should an unaccounted uncertainty in the ordinate be added? Defaults to :py:attr:`False`.
    """
    def __init__(self, temperature, diffusion, diffusion_error,
                 temperature_unit=UREG.kelvin,
                 diffusion_unit=UREG.centimeter**2 / UREG.second,
                 temperature_names=r'$T$',
                 diffusion_names=r'$D$', bounds=None, unaccounted_uncertainty=False):
        variable_names = [r'$E_a$', r'$A$']
        variable_units = [UREG.joules / UREG.mole, UREG.dimensionless]
        if unaccounted_uncertainty:
            variable_names.append('unaccounted uncertainty')
            variable_units.append(UREG.dimensionless)
        super().__init__(
            arrhenius, temperature, diffusion, diffusion_error, temperature_unit,
            diffusion_unit, temperature_names, diffusion_names, variable_names, variable_units, bounds, unaccounted_uncertainty)

    def sample(self, **kwargs):
        """
        Use MCMC to sample the posterior distribution of the relationship. For keyword arguments see the :func:`uravu.relationship.mcmc` docs. 
        """
        self.mcmc(prior_function=self.prior, **kwargs)

    def nested(self, **kwargs):
        """
        Use nested sampling to determine natural log-evidence for the model. For keyword arguments see the :func:`uravu.relationship.nested_sampling` docs.
        """
        self.nested_sampling(prior_function=self.prior, **kwargs)


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
        diffusion_error (:py:attr:`array_like`): Uncertainty in the diffusion coefficient data.
        temperature_unit (:py:class:`pint.unit.Unit`, optional): The unit for the temperature. Default is :py:attr:`kelvin`.
        diffusion_unit (:py:class:`pint.unit.Unit`, optional): The unit for the diffusion coefficient. Default is :py:attr:`centimetre**2 / second`.
        temperature_names (:py:attr:`str`, optional): The label for the temperature. Default is :py:attr:`$T$`.
        diffusion_names (:py:attr:`str`, optional): The label for the diffusion coefficient. Default is :py:attr:`$D$`.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters. Defaults to :py:attr:`None`.
        unaccounted_uncertainty (:py:attr:`bool`, optional): Should an unaccounted uncertainty in the ordinate be added? Defaults to :py:attr:`False`.
    """
    def __init__(self, temperature, diffusion, diffusion_error,
                 temperature_unit=UREG.kelvin,
                 diffusion_unit=UREG.centimeter**2 / UREG.second,
                 temperature_names=r'$T$',
                 diffusion_names=r'$D$', bounds=None, unaccounted_uncertainty=False):
        variable_names = [r'$E_a$', r'$A$', r'$T_0$']
        variable_units = [UREG.joules / UREG.mole, UREG.dimensionless, UREG.kelvin]
        if unaccounted_uncertainty:
            variable_names.append('unaccounted uncertainty')
            variable_units.append(UREG.dimensionless)
        super().__init__(
            super_arrhenius, temperature, diffusion, diffusion_error, temperature_unit,
            diffusion_unit, temperature_names, diffusion_names, variable_names, variable_units, bounds, unaccounted_uncertainty)

    def sample(self, **kwargs):
        """
        Use MCMC to sample the posterior distribution of the relationship. For keyword arguments see the :func:`uravu.relationship.mcmc` docs. 
        """
        self.mcmc(prior_function=self.prior, **kwargs)

    def nested(self, **kwargs):
        """
        Use nested sampling to determine natural log-evidence for the model. For keyword arguments see the :func:`uravu.relationship.nested_sampling` docs.
        """
        self.nested_sampling(prior_function=self.prior, **kwargs)

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
