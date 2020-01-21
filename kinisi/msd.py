"""
Class for MSD investigation.

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0913

from scipy.stats import linregress
from uncertainties import ufloat
from kinisi import utils, straight_line
from kinisi.distribution_array import DistributionArray


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
        self.data = None
        self.diffusion_coefficient = None

    def resample(self, **kwargs):
        """
        Resample the square-displacement data to obtain a description of
        the distribution as a function of delta_t.

        Args:
            data (list of array_like): A list of arrays, where
                each array has the axes [atom, squared displacement
                observation]. There is one array in the list for each
                delta_t value.
            ensure_normality (bool, optional): Use a Shapiro Wilks test to
                ensure that the distribution is normal for each element of the
                array.
            alpha (float, optional): Test metric for Shapiro-Wilks.
            n_resamples (int, optional): The number of resamples to be
                performed.
            samples_freq (int. optional): The frequency in observations to be
                sampled.
            confidence_interval (array_like): The percentile points of the
                distribution that should be stored.
        """
        self.data = utils.bootstrap(self.sq_displacements, **kwargs)

    def estimate_straight_line(self, delta_t):
        """
        Estimate the straight line gradient and intercept

        Args:
            delta_t (array_like): Values for abscissa.

        Returns:
            (array_like) Length of 2, gradient and intercept.
        """
        result = linregress(delta_t, self.data.medians)
        return result.slope, result.intercept

    def sample_diffusion(self, x_data):
        """
        Use MCMC sampling to evaluate diffusion coefficient.

        Args:
            x_data (array_like): Abscissa data.

        Returns:
        """
        dy_data = self.data.confidence_intervals[:, 1] - self.data.medians
        samples = straight_line.run_sampling(
            self.estimate_straight_line(x_data),
            self.data.medians,
            dy_data,
            x_data
        )
        straight_line_distro = DistributionArray(2, [2.5, 97.5])
        straight_line_distro.set_distribution(samples[:, 0], 0)
        straight_line_distro.set_distribution(samples[:, 1], 1)
        gradient = ufloat(
            straight_line_distro.medians[0],
            (straight_line_distro.confidence_intervals[
                0, 1] - straight_line_distro.medians[0])
        )
        self.diffusion_coefficient = gradient / 6
