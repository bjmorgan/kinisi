"""
The modules is focused on tools for the evaluation of the mean squared
displacement and resulting diffusion coefficient from a material.
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


def msd_bootstrap(displacements, n_resamples=1000, samples_freq=1,
                  confidence_interval=None, max_resamples=10000,
                  progress=True):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean
    and uncertainty for the squared displacements. This resampling method is
    applied until the MSD distribution is normal(or the `max_resamples` has
    been reached) and therefore may be described with a median and confidence
    interval.

    Args:
        displacements (list of array_like): A list of arrays, where
            each array has the axes [atom, displacement
            observation, dimension]. There is one array in the list for each
            delta_t value. Note: it is necessary to use a list of arrays
            as the number of observations is not necessary the same at
            each data point.
        n_resamples (int, optional): The initial number of resamples to
            be performed. Default is `1000`.
        samples_freq (int. optional): The frequency in observations to be
            sampled. Default is `1` (every observation).
        confidence_interval (array_like): The percentile points of the
            distribution that should be stored. Default is `[2.5, 97.5]` (a
            95 % confidence interval).
        max_resamples (int, optional): The max number of resamples to be
            performed by the distribution is assumed to be normal. This is
            necessary as for large numbers the Shapiro-Wilks test can be
            unstable. Default is `10000`.
        progress (bool, optional): Show tqdm progress for sampling. Default
            is `True`.

    Returns:
        (tuple of array_like) A tuple of two arrays, the first is the
            resampled mean squared displacement data, the second is the
            uncertainty in the mean squared displacement.
    """
    if confidence_interval is None:
        confidence_interval = [2.5, 97.5]
    max_obs = displacements[0].shape[1]
    mean_msd = np.zeros((len(displacements)))
    err_msd = np.zeros((len(displacements)))
    if progress:
        iterator = tqdm(range(len(displacements)))
    else:
        iterator = range(len(displacements))
    for i in iterator:
        d_squared = np.sum(displacements[i], axis=2) ** 2
        n_obs = displacements[i].shape[1]
        n_atoms = displacements[i].shape[0]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlap
        # Evaluate MSD first
        n_samples_msd = int(max_obs / dt_int * n_atoms / samples_freq)
        resampled = [
            np.mean(resample(d_squared.flatten(), n_samples=n_samples_msd))
            for j in range(n_resamples)
        ]
        distro = Distribution(
            resampled, "delta_t_{}".format(i), confidence_interval
        )
        while (
                not distro.normal) and distro.size < (
                    max_resamples-n_resamples):
            distro.add_samples(
                [
                    np.mean(
                        resample(
                            d_squared.flatten(),
                            n_samples=n_samples_msd)) for j in range(100)]
            )
        mean_msd[i] = distro.n
        err_msd[i] = (
            np.percentile(distro.samples, distro.ci_points[1]) - distro.n)
    return mean_msd, err_msd


def mscd_bootstrap(displacements, indices=None, n_resamples=1000,
                   samples_freq=1, confidence_interval=None,
                   max_resamples=10000, progress=True):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean
    and uncertainty for the squared charge displacements. This
    resampling method is applied until the MSCD distribution is normal(or the
    `max_resamples` has been reached) and therefore may be described with a
    median and confidence interval.

    Args:
        displacements (list of array_like): A list of arrays, where
            each array has the axes [atom, displacement
            observation, dimension]. There is one array in the list for each
            delta_t value. Note: it is necessary to use a list of arrays
            as the number of observations is not necessary the same at
            each data point.
        indices (array_like, optional): The indices of the particles of
            interest for charge displacement. Default to all particles.
        n_resamples (int, optional): The initial number of resamples to
            be performed. Default is `1000`.
        samples_freq (int. optional): The frequency in observations to be
            sampled. Default is `1` (every observation).
        confidence_interval (array_like): The percentile points of the
            distribution that should be stored. Default is `[2.5, 97.5]` (a
            95 % confidence interval).
        max_resamples (int, optional): The max number of resamples to be
            performed by the distribution is assumed to be normal. This is
            necessary as for large numbers the Shapiro-Wilks test can be
            unstable. Default is `10000`.
        progress (bool, optional): Show tqdm progress for sampling. Default
            is `True`.

    Returns:
        (tuple of array_like) A tuple of two arrays, the first is the
            resampled mean squared charge displacement data, the second is the
            uncertainty in the mean squared charge displacement.
    """
    if confidence_interval is None:
        confidence_interval = [2.5, 97.5]
    max_obs = displacements[0].shape[1]
    mean_mscd = np.zeros((len(displacements)))
    err_mscd = np.zeros((len(displacements)))
    if progress:
        iterator = tqdm(range(len(displacements)))
    else:
        iterator = range(len(displacements))
    for i in iterator:
        sq_chg_disp = np.sum(
            np.sum(displacements[i][indices, :, :], axis=0) ** 2, axis=1)
        n_obs = displacements[i].shape[1]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlap
        # Then evaluate MSCD
        n_samples_mscd = int(max_obs / dt_int / samples_freq)
        resampled = [
            np.mean(resample(sq_chg_disp.flatten(), n_samples=n_samples_mscd))
            for j in range(n_resamples)
        ]
        distro = Distribution(
            resampled, "delta_t_{}".format(i), confidence_interval
        )
        while (
                not distro.normal) and distro.size < (
                    max_resamples-n_resamples):
            distro.add_samples(
                [
                    np.mean(
                        resample(
                            sq_chg_disp.flatten(),
                            n_samples=n_samples_mscd)) for j in range(100)]
            )
        mean_mscd[i] = distro.n / len(indices)
        err_mscd[i] = (
            np.percentile(
                distro.samples,
                distro.ci_points[1]) - distro.n) / len(indices)
    return mean_mscd, err_mscd


class Diffusion(StraightLine):
    """
    Evaluate the diffusion coefficient from a set of (idelly) resampled MSD
    data and delta_t values.

    Attributes:
        diffusion_coefficient (uncertainties.ufloat or
            kinisi.distribution.Distribution): The value and associated
            uncertainty for the diffusion coeffcient for the MSD relationship
            with delta_t. The uncertainty is initially obtained from a
            weighted least squares fit, the accuracy of this can be improved
            by using the `sample()` method. The unit is
            centimeter ** 2 / second.
    """
    def __init__(self, delta_t, msd, msd_error,
                 delta_t_unit=UREG.femtoseconds, msd_unit=UREG.angstrom**2,
                 delta_t_names=r'$\delta t$',
                 msd_names=r'$\langle r ^ 2 \rangle$'):
        super().__init__(
            delta_t, msd, msd_error, delta_t_unit, msd_unit, delta_t_names,
            msd_names)
        self.diffusion_coefficient = self.variables[0] / 6 * (
            self.ordinate_unit / self.abscissa_unit)
        self.diffusion_coefficient = self.diffusion_coefficient.to(
            UREG.centimeter ** 2 / UREG.second)

    def sample(self, **kwargs):
        """
        Perform the MCMC sampling to obtain a more accurate description of the
        diffusion coefficient as a probability distribution.

        Keyword Args:
            walkers (int, optional): Number of MCMC walkers. Default is `100`.
            n_samples (int, optional): Number of sample points. Default is
                `500`.
            n_burn (int, optional): Number of burn in samples. Default is
                `500`.
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.
        """
        self.mcmc(**kwargs)
        unit_conversion = 1 * self.ordinate_unit / self.abscissa_unit
        self.diffusion_coefficient = Distribution(
            self.variables[0].samples * unit_conversion.to(
                UREG.centimeter ** 2 / UREG.second).magnitude / 6,
            name="$D$", unit=UREG.centimeter ** 2 / UREG.second)
