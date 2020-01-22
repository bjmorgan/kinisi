"""
DistributionArray class

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0902

import numpy as np
from scipy.stats import shapiro
from uncertainties import ufloat


class Distribution:
    """
    The Distribution class.

    Attributes:
        name (str):
        size (int): Number of samples in distribution.
        samples (array_like): Samples in the distribution.
        median (float): Median of distribution.
        error (float): Symetrical uncertainty on value, taken as 95 %
            confidence interval. `None` if distribution is not normal.
        ci_points (array_like): Percentiles to be stored as confidence
            interval.
        con_int (array_like): Confidence interval values.
        normal (bool): Distribution normally distributed.
    """
    def __init__(self, ci_points=None, name='Distribution'):
        """
        Args:
            ci_points (array_like, optional): The percentiles at which
                confidence intervals should be found.
        """
        self.name = name
        self.size = 0
        self.samples = np.array([])
        self.median = None
        self.error = None
        if ci_points is None:
            self.ci_points = [2.5, 97.5]
        else:
            self.ci_points = ci_points
        self.con_int = np.array([])
        self.normal = False

    def __repr__(self):
        """
        Custom representation.
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
            self.error = np.percentile(self.samples, 97.5) - self.median
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
