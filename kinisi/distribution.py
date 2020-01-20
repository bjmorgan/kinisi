"""
Distribution class

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

import numpy as np


class Distribution:
    """
    A class for describing distributions
    """
    def __init__(self, length, confidence_interval_points):
        """
        Args:
            length (int): Distribution length.
            confidence_interval_points (array_like): Percentiles to store
                in for each distribution point.
        """
        self.length = length
        self.median = np.zeros((self.length))
        self.confidence_interval_points = confidence_interval_points
        self.confidence_interval = np.zeros(
            (self.length, len(confidence_interval_points))
        )

    def set_value(self, median, confidence_interval, i):
        """
        Set the values of a point in the distribution.

        Args:
            median (float): Value of median of distribution point.
            confidence_interval (float): Value(s) of confidence interval
                points.
            i (int): value in distribution to set.
        """
        self.median[i] = median
        if len(confidence_interval) == len(self.confidence_interval_points):
            self.confidence_interval[i] = confidence_interval
        else:
            raise ValueError("Percentile should be array of "
                             "length {}".format(
                                 len(self.confidence_interval_points)))
