"""
The two classes herein enable the determination of the activation energy from a diffusion-Arrhenius or
-Super-Arrhenius plot.
This includes the uncertainty on the activation energy from the MCMC sampling of the plot, with uncertainties on
diffusion.
It is also easy to determine the Bayesian evidence for each of the models with the given data, enabling
the differentiation between data showing Arrhenius and Super-Arrhenius diffusion.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import Tuple, Union
import scipp as sc
from scipp.constants import R, N_A
from scipp.typing import VariableLike

R_eV = sc.to_unit(R / N_A, 'eV/K')

class TemperatureDependent: 
    """
    A base class for temperature-dependent relationships.
    This class will enable MCMC sampling of the temperature-dependent relationships and estimation of the Bayesian evidence.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances. 
    """
    def __init__(self, diffusion, function) -> 'TemperatureDependent':
        self.diffusion = diffusion
        self.temperature = diffusion.coords['temperature']
        self.function = function

        self.data_group = sc.DataGroup({'data': diffusion})

    def __repr__(self):
        """
        Return a string representation of the TemperatureDependent object.
        """
        return self.data_group.__repr__()
    
    def __str__(self):
        """
        Return a string representation of the TemperatureDependent object.
        """
        return self.data_group.__str__()
    
    def _repr_html_(self):
        """
        Return an HTML representation of the TemperatureDependent object.
        """
        return self.data_group._repr_html_()
    

def arrhenius(abscissa: VariableLike, activation_energy: VariableLike, prefactor: VariableLike) -> VariableLike:
    """
    Determine the diffusion coefficient for a given activation energy, and prefactor according to the Arrhenius
    equation.

    Args:
        abscissa (:py:attr:`array_like`): The abscissa data.
        activation_energy (:py:attr:`float`): The activation_energy value.
        prefactor (:py:attr:`float`): The prefactor value.

    :return: The diffusion coefficient data.
    """
    return prefactor * sc.exp(-1 * activation_energy / (R_eV * abscissa))


class Arrhenius(TemperatureDependent):
    """
    Evaluate the data with a standard Arrhenius relationship.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    """

    def __init__(self, diffusion, bounds=Union[Tuple[Tuple[float, float], Tuple[float, float]], None] = None) -> 'Arrhenius'  :
        super().__init__(diffusion, function=arrhenius)

    @property
    def activation_energy(self) -> sc.Variable:
        """
        :return: Activated energy distribution in electronvolt.
        """
        return self.data_group['data'].coords['activation_energy']

    @property
    def preexponential_factor(self) -> sc.Variable:
        """
        :return: Preexponential factor.
        """
        return self.data_group['data'].coords['preexponential_factor']
    
