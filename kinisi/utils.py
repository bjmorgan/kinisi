"""
Simple utility functions

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=W0127

import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import shapiro
from kinisi.distribution_array import DistributionArray


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


def logl(model, y_data, dy_data):
    """
    The natural logarithm of the joint likelihood, equation from
    DOI: 10.1107/S1600576718017296.

    Args:
        model (array_like): Model ordinate data.
        y_data (array_like): Experimental ordinate data.
        dy_data (array_like): Experimental ordinate-uncertainty data.
    """
    return -0.5 * np.sum(
        ((model - y_data) / dy_data) ** 2 + np.log(2 * np.pi * dy_data ** 2)
    )


def bootstrap(data, **kwargs):
    """
    Perform a bootstrap resampling.

    Args:
        data (list of array_like): A list of arrays, where
            each array has the axes [atom, squared displacement
            observation]. There is one array in the list for each
            delta_t value.
        ensure_normality (bool, optional): Use a Shapiro Wilks test to
            ensure that the distribution is normal for each element of the
            array.
        alpha (float, optional): Test metric for Shapiro-Wilks.
        n_resamples (int, optional): The number of resamples to be performed.
        samples_freq (int. optional): The frequency in observations to be
            sampled.
        confidence_interval (array_like): The percentile points of the
            distribution that should be stored.

    Returns:
        (kinisi.distribution.Distribution) The bootstrap determined
            distribution.
    """
    ensure_normality = True
    samples_freq = 1
    confidence_interval = [2.5, 97.5]
    alpha = 0.05
    n_resamples = 1000
    for key, value in kwargs.items():
        if key == 'ensure_normality':
            ensure_normality = value
        if key == 'samples_freq':
            samples_freq = value
        if key == 'confidence_interval':
            confidence_interval = value
        if key == 'alpha':
            alpha = value
        if key == 'n_samples':
            n_resamples = value
    max_obs = data[0].shape[1]
    distro = DistributionArray(len(data), confidence_interval)
    for i in tqdm(range(len(data))):
        disp = data[i]
        n_obs = disp.shape[1]
        n_atoms = disp.shape[0]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlap
        n_samples = int(max_obs / dt_int * n_atoms / samples_freq)
        sampled_means = [
            np.mean(
                resample(disp.flatten(), n_samples=n_samples)
            ) for j in range(n_resamples)
        ]
        if ensure_normality:
            while shapiro(
                    np.random.choice(
                        sampled_means,
                        size=n_resamples,
                        replace=False
                    )
            )[1] > alpha:
                sampled_means.append(
                    np.mean(resample(disp.flatten(), n_samples=n_samples))
                )
        distro.set_distribution(sampled_means, i)
    return distro
