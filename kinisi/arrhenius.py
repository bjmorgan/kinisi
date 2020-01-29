"""
Functions to enable the determination of the mean squared displacement of a
collection of atoms.
"""
# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913

from kinisi.distribution import Distribution
from kinisi.relationships import Arrhenius, VTFEquation
from . import UREG


class StandardArrhenius(Arrhenius):
    """
    The Activation energy
    """
    def __init__(self, temperature, diffusion, diffusion_error,
                 temperature_units=UREG.kelvin,
                 diffusion_units=UREG.centimeter**2 / UREG.second,
                 temperature_names=r'$T$',
                 diffusion_names=r'$D$'):
        super().__init__(
            temperature, diffusion, diffusion_error, temperature_units,
            diffusion_units, None, temperature_names, diffusion_names)

        self.activation_energy = self.variables[0] * UREG.joules
        self.activation_energy = self.activation_energy.to(UREG.electron_volt)

    def sample(self, **kwargs):
        """
        MCMC sampling
        """
        self.mcmc(**kwargs)
        unit_conversion = 1 * UREG.joule
        self.activation_energy = Distribution(
            self.variables[0].samples * unit_conversion.to(
                UREG.electron_volt).magnitude,
            name="$D$", units=UREG.electron_volt)


class SuperArrhenius(VTFEquation):
    """
    The Activation energy
    """
    def __init__(self, temperature, diffusion, diffusion_error,
                 temperature_units=UREG.kelvin,
                 diffusion_units=UREG.centimeter**2 / UREG.second,
                 temperature_names=r'$T$',
                 diffusion_names=r'$D$'):
        super().__init__(
            temperature, diffusion, diffusion_error, temperature_units,
            diffusion_units, None, temperature_names, diffusion_names)

        self.activation_energy = self.variables[0] * UREG.joules
        self.activation_energy = self.activation_energy.to(UREG.electron_volt)

    def sample(self, **kwargs):
        """
        MCMC sampling
        """
        self.mcmc(**kwargs)
        unit_conversion = 1 * UREG.joule
        self.activation_energy = Distribution(
            self.variables[0].samples * unit_conversion.to(
                UREG.electron_volt).magnitude,
            name="$D$", units=UREG.electron_volt)
