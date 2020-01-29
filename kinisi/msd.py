"""
Functions to enable the determination of the mean squared displacement of a
collection of atoms.
"""
# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913,R0914

import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
from kinisi.distribution import Distribution
from kinisi.relationships import StraightLine
from . import UREG


def bootstrap(data, n_resamples=1000, samples_freq=1,
              confidence_interval=None, progress=True):
    """
    Perform a bootstrap resampling.

    Args:
        data (list of array_like): A list of arrays, where
            each array has the axes [atom, displacement
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
        d_squared = data[i] ** 2
        n_obs = data[i].shape[1]
        n_atoms = data[i].shape[0]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlap
        n_samples = int(max_obs / dt_int * n_atoms / samples_freq)
        resampled = [
            np.mean(resample(d_squared.flatten(), n_samples=n_samples))
            for j in range(n_resamples)
        ]
        distro = Distribution(
            resampled, "delta_t_{}".format(i), confidence_interval
        )
        while not distro.normal:
            distro.add_samples(
                [np.mean(resample(d_squared.flatten(), n_samples=n_samples))]
            )
        mean_data[i] = distro.median
        err_data[i] = distro.error
    return mean_data, err_data


class Diffusion(StraightLine):
    """
    The mean squared displacement (MSD).
    """
    def __init__(self, delta_t, msd, msd_error,
                 delta_t_unit=UREG.femtoseconds, msd_unit=UREG.angstrom**2,
                 delta_t_names=r'$\delta t$',
                 msd_names=r'$\langle r ^ 2 \rangle$'):
        super().__init__(
            delta_t, msd, msd_error, delta_t_unit, msd_unit, None,
            delta_t_names, msd_names)
        self.diffusion_coefficient = self.variables[0] / 6 * (
            self.ordinate_unit / self.abscissa_unit)
        self.diffusion_coefficient = self.diffusion_coefficient.to(
            UREG.centimeter ** 2 / UREG.second)

    def sample(self, **kwargs):
        """
        MCMC sampling
        """
        self.mcmc(**kwargs)
        unit_conversion = 1 * self.ordinate_unit / self.abscissa_unit
        self.diffusion_coefficient = Distribution(
            self.variables[0].samples * unit_conversion.to(
                UREG.centimeter ** 2 / UREG.second).magnitude / 6,
            name="$D$", unit=UREG.centimeter ** 2 / UREG.second)
