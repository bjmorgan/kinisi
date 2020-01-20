"""
Simple utility functions

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=W0127

import numpy as np
from sklearn.utils import resample
from kinisi.distribution import Distribution


def straight_line(abscissa, gradient, intercept):
    """
    Calculate ordinate of straight line

    Args:
        abscissa (array_like): abscissa
        gradient (float): gradient
        intercept (float): intercept

    Returns:
        (array_like) ordinate
    """
    return gradient * abscissa + intercept


def bootstrap(data, resamples=2000, samples_freq=1, confidence_interval=None):
    """
    Perform a bootstrap resampling.

    Args:
        data (list of array_like): A list of arrays, where
            each array has the axes [atom, squared displacement
            observation]. There is one array in the list for each
            delta_t value.
        resamples (int, optional): The number of resamples to be performed.
        samples_freq (int. optional): The frequency in observations to be
            sampled.
        confidence_interval (array_like): The percentile points of the
            distribution that should be stored.

    Returns:
        (kinisi.distribution.Distribution) The bootstrap determined
            distribution.
    """
    max_obs = data[0].shape[1]
    if confidence_interval is None:
        confidence_interval = [2.5, 97.5]
    else:
        confidence_interval = confidence_interval
    distro = Distribution(len(data), confidence_interval)
    for i, disp in enumerate(data):
        n_obs = disp.shape[1]
        n_atoms = disp.shape[0]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlpa
        n_samples = int(max_obs / dt_int * n_atoms / samples_freq)
        sampled_means = [
            np.mean(
                resample(disp.flatten(), n_samples=n_samples)
            ) for j in range(resamples)
        ]
        # 2.5 %, 50 %, and 97.5 % percentile values for the mean squared
        # displacement at dt
        distro.set_value(
            np.percentile(sampled_means, 50.),
            np.array(
                [
                    np.percentile(
                        sampled_means,
                        j
                    ) for j in confidence_interval
                ]
            ),
            i,
        )
    return distro
