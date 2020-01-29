"""
This module contains a (small) library of relationships between the abscissa
and ordinate.
This includes the `StraightLine`, `Arrhenius`, and `VTFEquation`; all of which
are important in the study of diffusion in materials.
The `kinisi.relationships.Relationship` class is subclasses by these
relationships and enables the simple fitting of variables, the evalulation of
variable probability distributions by Markov-chain Monte Carlo (MCMC) and the
determination for the Bayesian evidence between a relationship and some data,
using nested sampling.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913,C0330,R0902

import numpy as np
import emcee
import dynesty
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
from scipy.constants import k
from uncertainties import ufloat
from kinisi import _fig_params
from kinisi.distribution import Distribution
from . import UREG


class Relationship:
    """
    The `Relationship` class enables the investigation of variable
    distributions in mathematical models.
    This is achieved by implementing simple fitting, MCMC

    Attributes:
        equation (function): A function outlining the relationship.
        abscissa (array_like): The abscissa data.
        ordinate (array_like): The ordinate data.
        ordinate_error (array_like): The uncertainty in the ordinate
            data.
        abscissa_unit (pint.UnitRegistry()): The unit for the
            abscissa.
        ordinate_unit (pint.UnitRegistry()): The unit for the
            ordinate.
        abscissa_name (str): The label for the abscissa.
        ordinate_name (str): The label for the ordinate.
        variables (lsit of uncertainties.ufloat or
            kinisi.distribution.Distribution): The variables associated
            with the mathematical relationship.
        variables_unit (list of pint.UnitRegistry()): The unit for the
            ordinate.
        variables_name (list of str): The label for the abscissa.
        evidence (float): The Bayesian evidence for the relationship with
        the data.
    """
    def __init__(self, equation, abscissa, ordinate, ordinate_error,
                 abscissa_unit=UREG.dimensionless,
                 ordinate_unit=UREG.dimensionless, abscissa_name="$x$",
                 ordinate_name="$y$"):
        """
        Args:
            abscissa (array_like): The abscissa data.
            ordinate (array_like): The ordinate data.
            ordinate_error (array_like): The uncertainty in the ordinate
                data.
            abscissa_unit (pint.UnitRegistry(), optional): The unit for the
                abscissa. Default is dimensionless.
            ordinate_unit (pint.UnitRegistry(), optional): The unit for the
                ordinate. Default is dimensionless.
            abscissa_name (str, optional): The label for the abscissa. Default
                is `$x$`.
            ordinate_name (str, optional): The label for the ordinate. Default
                is `$y$`.
        """
        self.equation = equation
        self.abscissa = abscissa
        self.ordinate = ordinate
        self.ordinate_error = ordinate_error
        self.abscissa_unit = abscissa_unit
        self.ordinate_unit = ordinate_unit
        self.abscissa_name = abscissa_name
        self.ordinate_name = ordinate_name

        self.variables = [None]
        self.variables_unit = [None]
        self.variables_name = [None]

        self.evidence = None

    def fit(self, init_values, with_uncertainty=True):
        """
        Perform a simple least squares fitting of the relationship to the
        data.

        Args:
            with_uncertainty (bool, optional): Should a weighted least
                squares be performed? Default is `True`.

        Returns:
            list of uncertainties.ufloat: The variables associated with the
                relationship.
        """
        if with_uncertainty:
            uncertainty = self.ordinate_error
        else:
            uncertainty = None
        popt, pcov = curve_fit(
            self.equation,
            self.abscissa, self.ordinate, p0=init_values, sigma=uncertainty)
        perr = np.sqrt(np.diag(pcov))
        variables = []
        for i, value in enumerate(popt):
            variables.append(ufloat(value, perr[i]))
        return variables

    def mcmc(self, walkers=100, n_samples=500, n_burn=500, progress=True):
        """
        Perform MCMC to get the probability distributions for the variables
        of the relationship. Note running this method will populate the
        `self.variables` attribute with `kinisi.distribution.Distribution`
        objects.

        Args:
            walkers (int, optional): Number of MCMC walkers. Default is `100`.
            n_samples (int, optional): Number of sample points. Default is
                `500`.
            n_burn (int, optional): Number of burn in samples. Default is
                `500`.
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.
        """
        uniform = np.random.uniform(size=(len(self.variables), walkers))
        initial_prior = self.prior_uniform(uniform).T
        ndims = initial_prior.shape[1]

        sampler = emcee.EnsembleSampler(walkers, ndims, self.comparison)

        sampler.run_mcmc(initial_prior, n_samples + n_burn, progress=progress)

        post_samples = sampler.get_chain(discard=n_burn).reshape((-1, ndims))

        for i in range(len(self.variables)):
            self.variables[i] = Distribution(
                post_samples[:, i],
                name=self.variables_name[i], unit=self.variables_unit[i])

    def nested_sampling(self, progress=True, **kwargs):
        """
        Perform the nested sampling in order to determine the Bayesian
        evidence.

        Args:
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.

        Keyword Args:
            See the `dynesty.run_nested()` documentation.
        """
        sampler = dynesty.NestedSampler(
            self.comparison, self.prior_uniform, len(self.variables))
        sampler.run_nested(print_progress=progress, **kwargs)
        results = sampler.results
        self.evidence = ufloat(results["logz"][-1], results["logzerr"][-1])

    def prior_uniform(self, uniform):
        """
        Generate the broad uniform priors.

        Args:
            uniform (array_like): An array of random uniform numbers (0, 1].
                The shape of which is M x N, where M is the number of
                parameters and N is the number of walkers.

        Returns:
            array_like: An array of random uniform numbers broadly distributed
                around the activation_energy and prefactor values from curve
                fitting.
        """
        broad = np.copy(uniform)
        for i, variable in enumerate(self.variables):
            broad[i] = norm.ppf(broad[i], loc=variable.n, scale=variable.s)
        return broad

    def plot(self, figsize=(10, 6)):
        """
        Plot the relationship. Additional plots will be included on this if
        the MCMC sampling has been used to find the activation_energy
        and prefactor distributions.

        Args:
            fig_size (tuple, optional): Horizontal and veritcal size for
                figure (in inches). Default is `(10, 6)`.

        Returns:
            (matplotlib.figure.Figure)
            (matplotlib.axes.Axes)
        """
        fig, axes = plt.subplots(figsize=figsize)
        axes.plot(self.abscissa, self.ordinate, c=list(_fig_params.TABLEAU)[0])
        x_label = "{}".format(self.abscissa_name)
        if self.abscissa_unit != UREG.dimensionless:
            x_label += "/${:~L}$".format(self.abscissa_unit)
        axes.set_xlabel(x_label)
        y_label = "{}".format(self.ordinate_name)
        if self.ordinate_unit != UREG.dimensionless:
            y_label += "/${:~L}$".format(self.ordinate_unit)
        axes.set_ylabel(y_label)
        axes.fill_between(
            self.abscissa,
            self.ordinate - self.ordinate_error,
            self.ordinate + self.ordinate_error,
            alpha=0.5, color=list(_fig_params.TABLEAU)[0])
        if not isinstance(self.variables[0], Distribution):
            variables = [var.n for var in self.variables]
            axes.plot(
                self.abscissa,
                self.equation(self.abscissa, *variables),
                color=list(_fig_params.TABLEAU)[1])
        else:
            plot_samples = np.random.randint(
                0, self.variables[0].samples.size, size=100)
            for i in plot_samples:
                variables = [var.samples[i] for var in self.variables]
                axes.plot(
                    self.abscissa,
                    self.equation(self.abscissa, *variables),
                    color=list(_fig_params.TABLEAU)[1], alpha=0.05)
        return fig, axes

    def comparison(self, theta):
        """
        Generate model data and get natural log of the likelihood.

        Args:
            theta (tuple): Values for variables.

        Returns:
            (float): ln-likelihood between model and data.
        """
        model = self.equation(self.abscissa, *theta)

        return lnl(model, self.ordinate, self.ordinate_error)


class StraightLine(Relationship):
    """
    A linear relationship defined by variables for the gradient (m) and
    intercept (c). Where y is the ordinate and x the abscissa.

    .. math::

       y = mx + c
    """
    def __init__(self, abscissa, ordinate, ordinate_error,
                 abscissa_unit=UREG.dimensionless,
                 ordinate_unit=UREG.dimensionless, variable_unit=None,
                 abscissa_name="$x$", ordinate_name="$y$",
                 variable_name=None):
        """
        Args:
            abscissa (array_like): The abscissa data.
            ordinate (array_like): The ordinate data.
            ordinate_error (array_like): The uncertainty in the ordinate
                data.
            abscissa_unit (pint.UnitRegistry(), optional): The unit for the
                abscissa. Default is dimensionless.
            ordinate_unit (pint.UnitRegistry(), optional): The unit for the
                ordinate. Default is dimensionless.
            variables_unit (list of pint.UnitRegistry(), optional): The
                unit for the variables. Default is
                `[ordinate_unit/abscissa_unit, ordinate_unit]`.
            abscissa_name (str): The label for the abscissa. Default is
                `$x$`.
            ordinate_name (str): The label for the ordinate. Default is
                `$y$`.
            variable_name (list of str): The label for the variables. Default
                is `[r'$m$', r'$c$']`.
        """
        super().__init__(
            straight_line, abscissa, ordinate, ordinate_error,
            abscissa_unit, ordinate_unit, abscissa_name, ordinate_name)
        if variable_unit is not None:
            self.variables_unit = variable_unit
        else:
            self.variables_unit = [
                ordinate_unit / abscissa_unit, ordinate_unit]
        if variable_name is not None:
            self.variables_name = variable_name
        else:
            self.variables_name = [r'$m$', r'$c$']

        results = linregress(self.abscissa, self.ordinate)
        init = [results.slope, results.intercept]
        self.variables = self.fit(init)


def straight_line(abscissa, gradient, intercept):
    """
    Determine the ordinate for a particular gradient, intercept and
    abscissa based on a linear model.

    Args:
        abscissa (array_like): The abscissa data.
        gradient (float): The gradient of the straight line.
        intercept (float): The y-intercept of the straight line.

    Returns:
        array_line: The ordinate data.
    """
    return gradient * abscissa + intercept


class Arrhenius(Relationship):
    r"""
    A relationship defined by the Arrhenius equation, with the variables of
    activation energy (E_a) and prefactor (A). Where D is the ordinate and
    T the abscissa.

    .. math::

       D = A\exp\bigg(\frac{-E_a}{kT}\bigg)

    Attributes:
        activation_energy (ufloat or kinisi.distribution.Distribution): The
            distribution of the activation energy.
    """
    def __init__(self, abscissa, ordinate, ordinate_error,
                 abscissa_unit=UREG.dimensionless,
                 ordinate_unit=UREG.dimensionless, variable_unit=None,
                 abscissa_name="$T$", ordinate_name="$k$",
                 variable_name=None):
        """
        Args:
            abscissa (array_like): The abscissa data.
            ordinate (array_like): The ordinate data.
            ordinate_error (array_like): The uncertainty in the ordinate
                data.
            abscissa_unit (pint.UnitRegistry(), optional): The unit for the
                abscissa. Default is dimensionless.
            ordinate_unit (pint.UnitRegistry(), optional): The unit for the
                ordinate. Default is dimensionless.
            variables_unit (list of pint.UnitRegistry(), optional): The
                unit for the variables. Default is `[dimensionless,
                ordinate_unit`].
            abscissa_name (str): The label for the abscissa. Default is
                `$T$`.
            ordinate_name (str): The label for the ordinate. Default is
                `$k$`.
            variable_name (list of str): The label for the variables. Default
                is `[r'$E_a$', r'$A$']`.
        """
        super().__init__(
            arrhenius, abscissa, ordinate, ordinate_error, abscissa_unit,
            ordinate_unit, abscissa_name, ordinate_name)
        if variable_unit is not None:
            self.variables_unit = variable_unit
        else:
            self.variables_unit = [UREG.dimensionless, ordinate_unit]
        if variable_name is not None:
            self.variables_name = variable_name
        else:
            self.variables_name = [r'$E_a$', r'$A$']

        results = linregress(1/self.abscissa, np.log(self.ordinate))
        init = [results.slope * k * -1, np.exp(results.intercept)]
        self.variables = self.fit(init)


def arrhenius(abscissa, activation_energy, prefactor):
    """
    Determine the diffusion coefficient for a given activation energy, and
    prefactor according to the Arrhenius equation.

    Args:
        abscissa (array_like): The abscissa data.
        activation_energy (float): The activation_energy value.
        prefactor (float): The prefactor value.

    Returns:
        array_line: The diffusion coefficient data.
    """
    return prefactor * np.exp(-1 * activation_energy / (k * abscissa))


class VTFEquation(Relationship):
    r"""
    A relationship defined by the Vogel–Tammann–Fulcher equation, with the
    variables of activation energy (E_a), prefactor (A), and temperature
    offset (T_0). Where D is the ordinate and T the abscissa.

    .. math::

       D = A \exp\bigg(\frac{-E_a}{k(T-T_0)}\bigg)

    Attributes:
        activation_energy (ufloat or kinisi.distribution.Distribution): The
            distribution of the activation energy.
    """
    def __init__(self, abscissa, ordinate, ordinate_error,
                 abscissa_unit=UREG.dimensionless,
                 ordinate_unit=UREG.dimensionless, variable_unit=None,
                 abscissa_name="$T$", ordinate_name="$k$",
                 variable_name=None):
        """
        Args:
            abscissa (array_like): The abscissa data.
            ordinate (array_like): The ordinate data.
            ordinate_error (array_like): The uncertainty in the ordinate
                data.
            abscissa_unit (pint.UnitRegistry(), optional): The unit for the
                abscissa. Default is dimensionless.
            ordinate_unit (pint.UnitRegistry(), optional): The unit for the
                ordinate. Default is dimensionless.
            variables_unit (list of pint.UnitRegistry(), optional): The
                unit for the variables. Default is `[dimensionless,
                ordinate_unit, abscissa_unit`].
            abscissa_name (str): The label for the abscissa. Default is
                `$T$`.
            ordinate_name (str): The label for the ordinate. Default is
                `$k$`.
            variable_name (list of str): The label for the variables. Default
                is `[r'$E_a$', r'$A$', r'$T_0$']`.
        """
        super().__init__(
            super_arrhenius, abscissa, ordinate, ordinate_error,
            abscissa_unit, ordinate_unit, abscissa_name, ordinate_name)
        if variable_unit is not None:
            self.variables_unit = variable_unit
        else:
            self.variables_unit = [
                UREG.dimensionless, ordinate_unit, abscissa_unit]
        if variable_name is not None:
            self.variables_name = variable_name
        else:
            self.variables_name = [r'$E_a$', r'$A$', r'$T_0$']

        results = linregress(1/self.abscissa, np.log(self.ordinate))
        init = [results.slope * k * -1, np.exp(results.intercept), 0]
        self.variables = self.fit(init)


def super_arrhenius(abscissa, activation_energy, prefactor, t_zero):
    """
    Determine the rate constant for a given activation energy, prefactor,
    and t_zero according to the Vogel–Tammann–Fulcher equation.

    Args:
        abscissa (array_like): The abscissa data.
        activation_energy (float): The activation_energy.
        prefactor (float): The prefactor.
        t_zero (float): The T_0 value.

    Returns:
        array_line: The ordinate data.
    """
    return prefactor * np.exp(
        -1 * activation_energy / (k * (abscissa - t_zero)))


def lnl(model, y_data, dy_data):
    """
    The natural logarithm of the joint likelihood, equation from
    DOI: 10.1107/S1600576718017296.

    Args:
        model (array_like): Model ordinate data.
        y_data (array_like): Experimental ordinate data.
        dy_data (array_like): Experimental ordinate-uncertainty data.

    Returns:
        (float): ln-likelihood between model and data.
    """
    return -0.5 * np.sum(
        ((model - y_data) / dy_data) ** 2 + np.log(2 * np.pi * dy_data ** 2))
