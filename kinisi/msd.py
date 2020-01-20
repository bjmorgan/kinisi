"""
Bootstrap resampler functions

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0913

from scipy.stats import linregress
from kinisi import utils


class MSD:
    """
    Mean-squared displacements, with uncertainties.
    """
    def __init__(self, sq_displacements, step_freq=1):
        """
        Args:
            sq_displacements (list of array_like): A list of arrays, where
                each array has the axes [atom, squared displacement
                observation] and describes the squared displacements.
                There is one array in the list for each delta_t value.
                Note: this can't be a 3D array because we have a different
                number of displacements at each dt value.
            step_freq (int, option): The frequency in delta_t to be sampled.
            confidence_interval (tuple): Percentiles to be determined in the
                distribution.
        """
        self.sq_displacements = sq_displacements[::step_freq]
        self.distribution = None

    def resample(self, resamples=2000, samples_freq=1,
                 confidence_interval=None):
        """
        Resample the square-displacement data to obtain a description of
        the distribution as a function of delta_t.

        Args:
            resamples (int, optional): The number of resamples to be performed.
            samples_freq (int. optional): The frequency in observations to be
                sampled.
            confidence_interval (array_like): The percentile points of the
                distribution that should be stored.
        """
        self.distribution = utils.bootstrap(
            self.sq_displacements,
            resamples,
            samples_freq,
            confidence_interval
        )

    def estimate_straight_line(self, delta_t):
        """
        Estimate the straight line gradient and intercept

        Args:
            delta_t (array_like): Values for abscissa.

        Returns:
            (array_like) Length of 2, gradient and intercept.
        """
        result = linregress(delta_t, self.distribution.median)
        return result.slope, result.intercept
