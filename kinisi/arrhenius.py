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

from collections.abc import Callable

import numpy as np
import scipp as sc
from emcee import EnsembleSampler
from scipp.constants import N_A, R
from scipp.typing import VariableLike
from scipy.linalg import pinvh
from scipy.optimize import minimize
from scipy.stats import uniform

from kinisi.samples import Samples

R_eV = sc.to_unit(R / N_A, 'eV/K')


class TemperatureDependent:
    """
    A base class for temperature-dependent relationships. This class will enable MCMC sampling of the
    temperature-dependent relationships and estimation of the Bayesian evidence.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    :param function: A callable function that describes the relationship between temperature and diffusion.
    :param bounds: Optional bounds for the parameters of the function. Defaults to None, in which case these
        are defined as +/- 50 percent of the best fit values.
    """

    def __init__(self, diffusion, function: Callable, bounds: None = None) -> 'TemperatureDependent':
        self.diffusion = diffusion
        self.temperature = diffusion.coords['temperature']
        self.data_group = sc.DataGroup({'data': diffusion})
        self.function = function
        self.bounds = bounds
        self.max_likelihood()
        if self.bounds is None:
            self.bounds = tuple([(p.value * 0.5 * p.unit, p.value * 1.5 * p.unit) for p in self._parameters])
        self.priors = [uniform(b[0].value, b[1].value - b[0].value) for b in self.bounds]

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

    def log_likelihood(self, parameters: tuple[float]) -> float:
        """
        Calculate the likelihood of the model given the data.

        :param model: The model to evaluate.
        :return: The likelihood of the model.
        """
        model = self.function(self.temperature.values, *parameters)

        covariance_matrix = np.diag(self.diffusion.variances)
        y_values = self.diffusion.values

        _, logdet = np.linalg.slogdet(covariance_matrix)
        logdet += np.log(2 * np.pi) * y_values.size
        inv = pinvh(covariance_matrix)

        diff = model - y_values
        logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
        return logl

    def nll(self, parameters: tuple[float]) -> float:
        """
        Calculate the negative log likelihood of the model given the data.

        :param parameters: The parameters of the model.

        :return: The negative log likelihood of the model.
        """
        return -self.log_likelihood(parameters)

    def log_prior(self, parameters: tuple[float]) -> float:
        """
        Calculate the log prior probability of the model parameters using a uniform prior based on the bounds.

        :param parameters: The parameters of the model.

        :return: The log prior probability of the model parameters.
        """
        return np.sum([self.priors[i].logpdf(parameters[i]) for i in range(len(parameters))])

    def log_posterior(self, parameters: tuple[float]) -> float:
        """
        Calculate the log posterior probability of the model parameters.

        :param parameters: The parameters of the model.

        :return: The log posterior probability of the model parameters.
        """
        return self.log_likelihood(parameters) + self.log_prior(parameters)

    def max_likelihood(self) -> tuple[float]:
        """
        Find the best fit parameters for the model.
        This method should be implemented in subclasses.
        """
        if self.bounds is not None:
            x0 = [((b[1] + b[0]) / 2) for b in self.bounds]
        else:
            x0 = [1 * u for u in self.parameter_units]
        result = minimize(self.nll, [x.value for x in x0]).x
        for i, name in enumerate(self.parameter_names):
            self.data_group[name] = result[i] * self.parameter_units[i]

    def mcmc(self, n_samples: int = 1000, n_walkers: int = 32, n_burn: int = 500, n_thin=10) -> None:
        """
        Perform MCMC sampling of the model parameters.
        This will sample the activation energy and preexponential factor.
        """
        if isinstance(self._parameters[0], Samples):
            values = np.array([sc.mean(p).value for p in self._parameters])
        else:
            values = np.array([p.value for p in self._parameters])
        pos = values + values * 1e-2 * np.random.randn(n_walkers, len(self._parameters))
        nwalkers, ndim = pos.shape

        sampler = EnsembleSampler(nwalkers, ndim, self.log_posterior)
        sampler.run_mcmc(pos, n_samples + n_burn, progress=True)
        flatchain = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)
        for i, name in enumerate(self.parameter_names):
            self.data_group[name] = Samples(flatchain[:, i], unit=self.parameter_units[i])

    @property
    def flatchain(self) -> sc.DataGroup:
        """
        :returns: The flatchain of the MCMC samples.
        """
        flatchain = {name: self.data_group[name] for name in self.parameter_names}
        return sc.DataGroup(**flatchain)


def arrhenius(abscissa: VariableLike, activation_energy: VariableLike, prefactor: VariableLike) -> VariableLike:
    """
    Determine the diffusion coefficient for a given activation energy, and prefactor according to the Arrhenius
    equation.

    :param abscissa: The temperature data.
    :param activation_energy: The activation_energy value.
    :param prefactor: The prefactor value.

    :return: The diffusion coefficient data.
    """
    return prefactor * np.exp(-1 * activation_energy / (R_eV.values * abscissa))


class Arrhenius(TemperatureDependent):
    """
    Evaluate the data with a standard Arrhenius relationship.

    :param diffusion: Diffusion coefficient sc.DataFrame with a temperature coordinate and variances.
    :param bounds: Optional bounds for the parameters of the function. Defaults to None, in which case these
        are defined as +/- 50 percent of the best fit values.
    """

    def __init__(
        self,
        diffusion,
        bounds: tuple[tuple[VariableLike, VariableLike], tuple[VariableLike, VariableLike]] | None = None,
    ) -> 'Arrhenius':
        self.parameter_names = ('activation_energy', 'preexponential_factor')
        self.parameter_units = (sc.Unit('eV'), sc.Unit('cm^2/s'))
        super().__init__(diffusion, function=arrhenius, bounds=bounds)

    @property
    def _parameters(self) -> tuple[VariableLike, VariableLike]:
        """
        :return: The parameters of the Arrhenius model.
        """
        return (self.activation_energy, self.preexponential_factor)

    @property
    def activation_energy(self) -> VariableLike | Samples:
        """
        :return: Activated energy distribution in electronvolt.
        """
        return self.data_group['activation_energy']

    @property
    def preexponential_factor(self) -> VariableLike | Samples:
        """
        :return: Preexponential factor.
        """
        return self.data_group['preexponential_factor']
