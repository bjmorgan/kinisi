"""
Simple utility functions

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0913

import numpy as np
from tqdm import tqdm
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


def lnl(model, y_data, dy_data):
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


def bootstrap(data, n_resamples=1000, samples_freq=1,
              confidence_interval=None, progress=True):
    """
    Perform a bootstrap resampling.

    Args:
        data (list of array_like): A list of arrays, where
            each array has the axes [atom, squared displacement
            observation]. There is one array in the list for each
            delta_t value.
        n_resamples (int, optional): The initial number of resamples to
            be performed.
        samples_freq (int. optional): The frequency in observations to be
            sampled.
        confidence_interval (array_like): The percentile points of the
            distribution that should be stored.
        progress (bool, optional): Show tqdm progress for sampling.

    Returns:
        (tuple of array_like) A tuple of two arrays, the first is the
            resampled mean data and the second is the uncertainty on that
            data.
    """
    if confidence_interval is None:
        confidence_interval = [2.5, 97.5]
    max_obs = data[0].shape[1]
    mean_data = np.zeros((len(data)))
    err_data = np.zeros((len(data)))
    if progress:
        iterator = tqdm(range(len(data)))
    else:
        iterator = range(len(data))
    for i in iterator:
        n_obs = data[i].shape[1]
        n_atoms = data[i].shape[0]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlap
        n_samples = int(max_obs / dt_int * n_atoms / samples_freq)
        distro = Distribution(confidence_interval, name='delta_t_{}'.format(i))
        distro.add_samples(
            [
                np.mean(
                    resample(data[i].flatten(), n_samples=n_samples)
                ) for j in range(n_resamples)
            ]
        )
        while not distro.normal:
            distro.add_samples(
                [np.mean(resample(data[i].flatten(), n_samples=n_samples))]
            )
        mean_data[i] = distro.median
        err_data[i] = distro.error
    return mean_data, err_data
