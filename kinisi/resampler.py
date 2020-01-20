"""
Bootstrap resampler functions

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0913

import numpy as np
from sklearn.utils import resample
from scipy.optimize import curve_fit
from kinisi import utils


class Distribution:
    """
    The distribution of displacements as a function of delta_t.
    """
    def __init__(self, displacements, resamples=2000, samples_freq=1,
                 step_freq=1, confidence_interval=None):
        """
        Args:
            displacements (list of array_like): A list of arrays, where
                each array has the axes [atom, squared displacement
                observation]. There is one array in the list for each
                delta_t value. Note: this can't be a 3D array because
                we have a different number of displacements at each dt value.
            resamples (int, optional): The number of resamples to be performed.
            samples_freq (int. optional): The frequency in observations to be
                sampled.
            step_freq (int, option): The frequency in delta_t to be sampled.
            confidence_interval (tuple): Percentiles to be determined in the
            distribution.
        """
        self.displacements = displacements[::step_freq]
        self.resamples = resamples
        self.samples_freq = samples_freq
        self.median = np.zeros((len(displacements[::step_freq])))
        if confidence_interval is None:
            self.confidence_interval = [2.5, 97.5]
        else:
            self.confidence_interval = confidence_interval
        self.span = np.zeros(
            (
                len(displacements[::step_freq]),
                len(self.confidence_interval)
            )
        )

    def resample(self):
        """
        Perform a bootstrap resampling.
        """
        max_obs = self.displacements[0].shape[1]
        for i, disp in enumerate(self.displacements):
            n_obs = disp.shape[1]
            n_atoms = disp.shape[0]
            dt_int = max_obs - n_obs + 1
            # approximate number of "non-overlapping" observations, allowing
            # for partial overlpa
            n_samples = int(max_obs / dt_int * n_atoms / self.samples_freq)
            sampled_means = [
                np.mean(
                    resample(disp.flatten(), n_samples=n_samples)
                ) for j in range(self.resamples)
            ]
            # 2.5 %, 50 %, and 97.5 % percentile values for the mean squared
            # displacement at dt
            self.median[i] = np.percentile(sampled_means, 50.)
            self.span[i] = np.array(
                [
                    np.percentile(
                        sampled_means,
                        j
                    ) for j in self.confidence_interval
                ]
            )

    def estimate_straight_line(self, delta_t):
        """
        Estimate the straight line gradient and intercept

        Args:
            delta_t (array_like): Values for abscissa.

        Returns:
            (array_like) Lenght of 2, gradient and intercept.
        """
        popt = curve_fit(utils.straight_line, delta_t, self.median)[0]
        return popt
