"""
The Distribution class enables easier handling in probability distributions
in kinisi.
In addition to a helpful storage container for information about the
probability distribution, there is also helper functions to check the
normality of the distribution and create publication quality plots.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0902

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, gaussian_kde
from uncertainties import ufloat
from kinisi import _fig_params


class Distribution:
    """
    The Distribution class.

    Attributes:
        name (str): A name for the distrition.
        size (int): Number of samples in distribution.
        samples (array_like): Samples in the distribution.
        median (float): Median of distribution.
        error (float): Symetrical uncertainty on value, taken as 95 %
            confidence interval. `None` if distribution is not normal.
        ci_points (tuple): A tuple of two. The percentiles to be stored as
            confidence interval.
        con_int (array_like): Confidence interval values.
        normal (bool): Distribution normally distributed.
    """
    def __init__(self, name='Distribution', ci_points=None, units=None):
        """
        Args:
            name (str, optional): A name to identify the distribution.
            ci_points (array_like, optional): The percentiles at which
                confidence intervals should be found.
            units (pint.UnitRegistry(), optional) The units for the
                distribution. Default is `None`.
        """
        self.name = name
        self.size = 0
        self.samples = np.array([])
        self.median = None
        self.error = None
        if ci_points is None:
            self.ci_points = [2.5, 97.5]
        else:
            if len(ci_points) != 2:
                raise ValueError("The ci_points must be an array or tuple "
                                 "of length two.")
            self.ci_points = ci_points
        self.con_int = np.array([])
        self.normal = False
        self.units = units

    def __repr__(self):  # pragma: no cover
        """
        Custom repr.
        """
        return self.__str__()

    def __str__(self):  # pragma: no cover
        """
        Custom string.
        """
        representation = 'Distribution: {}\nSize: {}\n'.format(
            self.name,
            self.size,
        )
        representation += 'Samples: '
        if self.size > 5:
            representation += '[{} {} ... {} {}]\n'.format(
                self.samples[0],
                self.samples[1],
                self.samples[-2],
                self.samples[-1],
            )
        else:
            representation += '['
            representation += ' '.join(['{}'.format(i) for i in self.samples])
            representation += ']\n'
        representation += 'Median: {}\n'.format(self.median)
        if self.check_normality():
            representation += 'Symetrical Error: {}\n'.format(self.error)
        representation += 'Confidence intervals: ['
        representation += ' '.join(['{}'.format(i) for i in self.con_int])
        representation += ']\n'
        representation += 'Confidence interval points: ['
        representation += ' '.join(['{}'.format(i) for i in self.ci_points])
        representation += ']\n'
        if self.median is not None:
            representation += 'Reporting Value: '
            if self.check_normality():
                uncertainty = self.con_int[1] - self.median
                representation += "{}\n".format(
                    ufloat(self.median, uncertainty)
                )
            else:
                representation += "{}+{}-{}\n".format(
                    self.median,
                    self.con_int[1],
                    self.con_int[0],
                )
        representation += 'Normal: {}\n'.format(self.normal)
        return representation

    def check_normality(self):
        """
        Assess if the samples are normally distributed using a Shapiro-Wilks
        test.
        """
        samples = np.copy(self.samples)
        if self.size <= 3:
            self.normal = False
            self.error = None
            return False
        if self.size >= 5000:
            samples = np.random.choice(self.samples, size=2500, replace=False)
        alpha = 0.05
        p_value = shapiro(samples)[1]
        if p_value > alpha:
            self.normal = True
            self.error = np.percentile(
                self.samples,
                self.ci_points[1],
            ) - self.median
            return True
        self.normal = False
        self.error = None
        return False

    def add_samples(self, samples):
        """
        Add samples to the distribution.

        Args:
            samples (array_like): Samples to be added to the distribution.
        """
        self.samples = np.append(self.samples, np.array(samples).flatten())
        self.size = self.samples.size
        self.median = np.percentile(self.samples, 50.)
        if self.size > 1:
            self.con_int = np.array(
                [np.percentile(self.samples, i) for i in self.ci_points]
            )
        self.check_normality()

    def plot(self, figsize=(10, 6)):  # pragma: no cover
        """
        Plot the probability density function for the distribution.

        Args:
            fig_size (tuple): Horizontal and veritcal size for figure
                (in inches).

        Returns:
            (matplotlib.figure.Figure)
            (matplotlib.axes.Axes)
        """
        fig, axes = plt.subplots(figsize=figsize)
        kde = gaussian_kde(self.samples)
        abscissa = np.linspace(self.samples.min(), self.samples.max(), 100)
        ordinate = kde.evaluate(abscissa)
        axes.plot(
            abscissa,
            ordinate,
            color=list(_fig_params.TABLEAU)[0],
        )
        axes.hist(
            self.samples,
            bins=25,
            density=True,
            color=list(_fig_params.TABLEAU)[0],
            alpha=0.5,
        )
        axes.fill_betweenx(
            np.linspace(0, ordinate.max() + ordinate.max() * 0.1),
            self.con_int[0],
            self.con_int[1],
            alpha=0.2,
        )
        x_label = '{}'.format(self.name)
        if self.units:
            x_label += '/${:~L}$'.format(self.units)
        axes.set_xlabel(x_label)
        axes.set_ylabel('$p(${}$)$'.format(self.name))
        axes.set_ylim((0, ordinate.max() + ordinate.max() * 0.1))
        return fig, axes
