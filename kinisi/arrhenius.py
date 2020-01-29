"""
The two classes herein enable the determination of the activation energy from
a diffusion-Arrhenius or -Super-Arrhenius plot.
This includes the uncertainty on the activation energy from the MCMC sampling
of the plot, with uncertainties on diffusion.
Furthermore, the classes are build on the Relationship subclass, therefore
it is easy to determine the Bayesian evidence for each of the models with the
given data, enabling the distinction between Arrhenius and Super-Arrhenius
diffusion.
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
    Evaluate the data with a standard Arrhenius relationship.

    Attributes:
        activation_energy (uncertainties.ufloat or
            kinisi.distribution.Distribution): The value and associated
            uncertainty for the activation energy from the standard
            Arrhenius relationship. The uncertainty is initial obtained
            from a weighted least squares fit, the accuracy of this can
            be improved by using this `sample()` method. The units are eV.
    """
    def __init__(self, temperature, diffusion, diffusion_error,
                 temperature_units=UREG.kelvin,
                 diffusion_units=UREG.centimeter**2 / UREG.second,
                 temperature_names=r'$T$',
                 diffusion_names=r'$D$'):
        """
        Args:
            temperature (array_like): The temperature data.
            diffusion (array_like): The diffusion coefficient data.
            diffusion_error (array_like): The uncertainty in the diffusion
                coefficient data.
            temperature_units (pint.UnitRegistry(), optional): The units for
                the temperature. Default is kelvin.
            diffusion_units (pint.UnitRegistry(), optional): The units for
                the diffusion coefficient. Default is centimetre**2 per
                second.
            temperature_names (str): The label for the temperature. Default is
                `$T$`.
            diffusion_names (str): The label for the diffusion coefficient.
                Default is `$D$`.
        """
        super().__init__(
            temperature, diffusion, diffusion_error, temperature_units,
            diffusion_units, None, temperature_names, diffusion_names)

        self.activation_energy = self.variables[0] * UREG.joules
        self.activation_energy = self.activation_energy.to(UREG.electron_volt)

    def sample(self, **kwargs):
        """
        Perform the MCMC sampling to obtain a more accurate description of the
        activation energy as a probability distribution.
        """
        self.mcmc(**kwargs)
        unit_conversion = 1 * UREG.joule
        self.activation_energy = Distribution(
            self.variables[0].samples * unit_conversion.to(
                UREG.electron_volt).magnitude,
            name="$D$", units=UREG.electron_volt)


class SuperArrhenius(VTFEquation):
    """
    Evaluate the data with a Super-Arrhenius relationship.

    Attributes:
        activation_energy (uncertainties.ufloat or
            kinisi.distribution.Distribution): The value and associated
            uncertainty for the activation energy from the
            Super-Arrhenius relationship. The uncertainty is initial obtained
            from a weighted least squares fit, the accuracy of this can
            be improved by using this `sample()` method. The units are eV.
    """
    def __init__(self, temperature, diffusion, diffusion_error,
                 temperature_units=UREG.kelvin,
                 diffusion_units=UREG.centimeter**2 / UREG.second,
                 temperature_names=r'$T$',
                 diffusion_names=r'$D$'):
        """
        Args:
            temperature (array_like): The temperature data.
            diffusion (array_like): The diffusion coefficient data.
            diffusion_error (array_like): The uncertainty in the diffusion
                coefficient data.
            temperature_units (pint.UnitRegistry(), optional): The units for
                the temperature. Default is kelvin.
            diffusion_units (pint.UnitRegistry(), optional): The units for
                the diffusion coefficient. Default is centimetre**2 per
                second.
            temperature_names (str): The label for the temperature. Default is
                `$T$`.
            diffusion_names (str): The label for the diffusion coefficient.
                Default is `$D$`.
        """
        super().__init__(
            temperature, diffusion, diffusion_error, temperature_units,
            diffusion_units, None, temperature_names, diffusion_names)

        self.activation_energy = self.variables[0] * UREG.joules
        self.activation_energy = self.activation_energy.to(UREG.electron_volt)

    def sample(self, **kwargs):
        """
        Perform the MCMC sampling to obtain a more accurate description of the
        activation energy as a probability distribution.
        """
        self.mcmc(**kwargs)
        unit_conversion = 1 * UREG.joule
        self.activation_energy = Distribution(
            self.variables[0].samples * unit_conversion.to(
                UREG.electron_volt).magnitude,
            name="$D$", units=UREG.electron_volt)
