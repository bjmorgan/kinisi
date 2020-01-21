"""
DistributionArray class

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

import numpy as np
from scipy.stats import shapiro


class DistributionArray:
    """
    A class for describing an array of distributions. Note: the
    distributions must be a uniform size.
    """
    def __init__(self, length, confidence_interval_points):
        """
        Args:
            length (int): Array length.
            size (int): Size of distribution.
            confidence_interval_points (array_like): Percentiles to store
                in for each distribution point.
        """
        self.length = length
        self.distributions = [[]] * self.length
        self.medians = np.zeros((self.length))
        self.confidence_interval_points = confidence_interval_points
        self.confidence_intervals = np.zeros(
            (self.length, len(confidence_interval_points))
        )

    def set_distribution(self, samples, i):
        """
        Set a distribution in the array.

        Args:
            samples (array): Value of median of distribution point.
            confidence_interval (float): Value(s) of confidence interval
                points.
            i (int): distribution in array to set.
        """
        self.medians[i] = np.percentile(samples, 50.)
        self.confidence_intervals[i] = [
            np.percentile(samples, j) for j in self.confidence_interval_points
        ]
        self.distributions[i] = samples

    def check_normality(self, i):
        """
        Check if an a distribution in the array is normal

        Args:
            i (int): Distribution to check.

        Returns:
            (bool): True if the distribution is normal.
        """
        distribution = self.distributions[i]
        if len(distribution) > 5000:
            distribution = np.random.choice(
                distribution,
                size=5000,
                replace=False
            )
        p_value = shapiro(distribution)[1]
        alpha = 0.05
        return p_value > alpha
