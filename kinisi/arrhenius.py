"""
The two classes herein enable the determination of the activation energy from a diffusion-Arrhenius or
-Super-Arrhenius plot.
This includes the uncertainty on the activation energy from the MCMC sampling of the plot, with uncertainties on
diffusion.
Furthermore, the classes are build on the Relationship subclass, therefore it is easy to determine the Bayesian
evidence for each of the models with the given data, enabling the distinction between Arrhenius and Super-Arrhenius
diffusion.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import List, Union, Tuple
import numpy as np
from uravu.relationship import Relationship
from scipy.constants import R, N_A, eV

# Convert R to eV
R_no_mol = R / N_A
R_eV = R_no_mol / eV


class StandardArrhenius(Relationship):
    """
    Evaluate the data with a standard Arrhenius relationship.
    For attributes associated with the :py:class:`uravu.relationship.Relationship` class see that documentation.

    Args:
        temperature (:py:attr:`array_like`): Temperature data in kelvin.
        diffusion (:py:attr:`array_like`): Diffusion coefficient data in cm^2s^{-1}.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters. Defaults to :py:attr:`((0, 1), (0, 1e20))`.
        diffusion_error (:py:attr:`array_like`): Uncertainty in the diffusion coefficient data. Not necessary
            if :py:attr:`diffusion` is :py:attr:`list` of :py:class:`uravu.distribution.Distribution` objects.
        ci_points (:py:attr:`array_like`, optional): The two percentiles at which confidence intervals should be
            found for the variables. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
    """
    def __init__(self,
                 temperature: np.ndarray,
                 diffusion: Union[List['uravu.distribution.Distribution'], np.ndarray],
                 bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1e20)),
                 diffusion_error: np.ndarray = None,
                 ci_points: Tuple[float, float] = None):
        super().__init__(arrhenius, temperature, diffusion, bounds, ordinate_error=diffusion_error, ci_points=None)

    @property
    def activation_energy(self) -> 'uravu.distribution.Distribution':
        """
        :return: Activated energy distribution in electronvolt.
        """
        return self.variables[0]

    @property
    def preexponential_factor(self) -> 'uravu.distribution.Distribution':
        """
        :return: Preexponential factor.
        """
        return self.variables[1]


def arrhenius(abscissa: np.ndarray, activation_energy: float, prefactor: float) -> np.ndarray:
    """
    Determine the diffusion coefficient for a given activation energy, and prefactor according to the Arrhenius
    equation.

    Args:
        abscissa (:py:attr:`array_like`): The abscissa data.
        activation_energy (:py:attr:`float`): The activation_energy value.
        prefactor (:py:attr:`float`): The prefactor value.

    :return: The diffusion coefficient data.
    """
    return prefactor * np.exp(-1 * activation_energy / (R_eV * abscissa))


class SuperArrhenius(Relationship):
    """
    Evaluate the data with a super-Arrhenius relationship. For attributes associated with the
    :py:class:`uravu.relationship.Relationship` class see that documentation.

    Args:
        temperature (:py:attr:`array_like`): Temperature data in kelvin.
        diffusion (:py:attr:`array_like`): Diffusion coefficient data in cm^2s^{-1}.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters. Defaults to 
            :py:attr:`[(0, 1), (0, 1e20), (0, temperature[0])]`.
        diffusion_error (:py:attr:`array_like`): Uncertainty in the diffusion coefficient data. Not necessary
            if :py:attr:`diffusion` is :py:attr:`list` of :py:class:`uravu.distribution.Distribution` objects.
        ci_points (:py:attr:`array_like`, optional): The two percentiles at which confidence intervals should be
            found for the variables. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
    """
    def __init__(self,
                 temperature: np.ndarray,
                 diffusion: Union[List['uravu.distribution.Distribution'], np.ndarray],
                 bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = [(0, 1), (0, 1e20),
                                                                                                 (0, None)],
                 diffusion_error: np.ndarray = None,
                 ci_points: Tuple[float, float] = None):
        if bounds[2][1] is None:
            bounds[2] = (0, temperature[0])
        super().__init__(super_arrhenius,
                         temperature,
                         diffusion,
                         bounds,
                         ordinate_error=diffusion_error,
                         ci_points=None)

    @property
    def activation_energy(self) -> 'uravu.distribution.Distribution':
        """
        :return: Activated energy distribution in electronvolt.
        """
        return self.variables[0]

    @property
    def preexponential_factor(self) -> 'uravu.distribution.Distribution':
        """
        :return: Preexponential factor.
        """
        return self.variables[1]

    @property
    def T0(self) -> 'uravu.distribution.Distribution':
        """
        :return: Temperature factor for the VTF equation in kelvin.
        """
        return self.variables[2]


def super_arrhenius(abscissa: np.ndarray, activation_energy: float, prefactor: float, t_zero: float) -> np.ndarray:
    """
    Determine the rate constant for a given activation energy, prefactor, and t_zero according to the
    Vogel–Tammann–Fulcher equation.

    :param abscissa: The abscissa data.
    :param activation_energy: The activation_energy value.
    :param prefactor: The prefactor value.
    :param t_zero: The T_0 value.

    :return: The diffusion coefficient data.
    """
    return prefactor * np.exp(-1 * activation_energy / (R_eV * (abscissa - t_zero)))
