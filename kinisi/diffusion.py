"""
The modules is focused on tools for the evaluation of the mean squared displacement and resulting diffusion coefficient from a material.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913,R0914

import sys
import warnings
import numpy as np
from scipy.stats import uniform, norm
from sklearn.utils import resample
from tqdm import tqdm
from uravu.distribution import Distribution
from uravu.relationship import Relationship
from uravu import UREG, utils


def msd_bootstrap(delta_t, disp_3d, n_resamples=1000, samples_freq=1,
                  confidence_interval=None, max_resamples=100000,
                  bootstrap_multiplier=1, progress=True):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the squared displacements. 
    This resampling method is applied until the MSD distribution is normal(or the `max_resamples` has been reached) and therefore may be described with a median and confidence interval.

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values.
        displacements (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point.
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`.
        samples_freq (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation).
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`.
        bootstrap_multiplier (:py:attr:`int`, optional): The factor by which the number of bootstrap samples should be multiplied. The default is :py:attr:`1`, which is the maximum number of truely independent samples in a given timestep. This can be increase, however it is importance to note that when greater than 1 the sampling is no longer independent.
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`.

    Returns:
        :py:attr:`tuple`: Containing:
            - :py:attr:`array_like`: Timestep values that were resampled.
            - :py:attr:`array_like`: Resampled mean squared displacement.
            - :py:attr:`array_like`: Variance in mean squared displacement.
            - :py:attr:`list` of :py:class:`uravu.distribution.Distribution`: Resampled mean squared distributions.
    """
    if confidence_interval is None:
        confidence_interval = [2.5, 97.5]
    displacements = disp_3d[::samples_freq]
    delta_t = delta_t[::samples_freq]
    max_obs = displacements[0].shape[1]
    output_delta_t = np.array([])
    mean_msd = np.array([])
    err_msd = np.array([])
    con_int_msd_lower = np.array([])
    con_int_msd_upper = np.array([])
    distributions = []
    if progress:
        iterator = tqdm(range(len(displacements)), desc='Bootstrapping displacements')
    else:
        iterator = range(len(displacements))
    for i in iterator:
        d_squared = np.sum(displacements[i] ** 2, axis=2)
        n_obs = displacements[i].shape[1]
        n_atoms = displacements[i].shape[0]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlap
        # Evaluate MSD first
        n_samples_msd = int(
            max_obs / dt_int * n_atoms) * bootstrap_multiplier
        if n_samples_msd <= 1:
            continue
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
        if distro.size >= (max_resamples-n_resamples):
            warnings.warn("The maximum number of resamples has been reached, "
                          "and the distribution is not yet normal. The "
                          "distribution will be treated as normal.")
        output_delta_t = np.append(output_delta_t, delta_t[i])
        mean_msd = np.append(mean_msd, distro.n)
        err_msd = np.append(
            err_msd, np.var(distro.samples, ddof=1))
        distributions.append(distro)
    return (
        output_delta_t, mean_msd, err_msd, distributions)


def mscd_bootstrap(delta_t, disp_3d, indices=None, n_resamples=1000,
                   samples_freq=1, confidence_interval=None,
                   max_resamples=100000, bootstrap_multiplier=1,
                   progress=True):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the squared charge displacements. This resampling method is applied until the MSCD distribution is normal (or the `max_resamples` has been reached) and therefore may be described with a median and confidence interval.

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values.
        displacements (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point.
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`.
        samples_freq (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation).
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`.
        bootstrap_multiplier (:py:attr:`int`, optional): The factor by which the number of bootstrap samples should be multiplied. The default is :py:attr:`1`, which is the maximum number of truely independent samples in a given timestep. This can be increase, however it is importance to note that when greater than 1 the sampling is no longer independent.
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`.

    Returns:
        :py:attr:`tuple`: Containing:
           - :py:attr:`array_like`: Timestep values that were resampled.
            - :py:attr:`array_like`: Resampled mean squared charge displacement.
            - :py:attr:`array_like`: Variance in mean squared charge displacement.
            - :py:attr:`list` of :py:class:`uravu.distribution.Distribution`: Resampled mean squared charge distributions.
    """
    if confidence_interval is None:
        confidence_interval = [2.5, 97.5]
    displacements = disp_3d[::samples_freq]
    delta_t = delta_t[::samples_freq]
    max_obs = displacements[0].shape[1]
    output_delta_t = np.array([])
    mean_mscd = np.array([])
    err_mscd = np.array([])
    con_int_mscd_lower = np.array([])
    con_int_mscd_upper = np.array([])
    distributions = []
    if progress:
        iterator = tqdm(range(len(displacements)), desc='Bootstrapping displacements')
    else:
        iterator = range(len(displacements))
    for i in iterator:
        sq_com_motion = np.sum(displacements[i][indices, :, :], axis=0) ** 2
        sq_chg_disp = np.sum(sq_com_motion, axis=1)
        n_obs = displacements[i].shape[1]
        dt_int = max_obs - n_obs + 1
        # approximate number of "non-overlapping" observations, allowing
        # for partial overlap
        # Then evaluate MSCD
        n_samples_mscd = int(
            max_obs / dt_int / samples_freq) * bootstrap_multiplier
        if n_samples_mscd <= 1:
            continue
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
        if distro.size >= (max_resamples-n_resamples):
            warnings.warn("The maximum number of resamples has been reached, "
                          "and the distribution is not yet normal. The "
                          "distribution will be treated as normal.")
        output_delta_t = np.append(output_delta_t, delta_t[i])
        mean_mscd = np.append(mean_mscd, distro.n / len(indices))
        err_mscd = np.append(err_mscd, np.var(distro.samples, ddof=1))
        distributions.append(distro)
    return (
        output_delta_t, mean_mscd, err_mscd, distributions)


class Diffusion(Relationship):
    r"""
    Evaluate the data with a Einstein diffusion relationship. 
    For attributes associated with the :py:class:`uravu.relationship.Relationship` class see that documentation.
    This :py:attr:`uravu.relationship.Relationship.variables` for this model is a :py:attr:`list` of length 2, where :py:attr:`~uravu.relationship.Relationship.variables[0]` is the gradient of the straight line and :py:attr:`~uravu.relationship.Relationship.variables[1]` is the offset of the ordinate. 

    Args:       
        delta_t (:py:attr:`array_like`): Timestep data.
        msd (:py:attr:`array_like`): Mean squared displacement data.
        bounds (:py:attr:`tuple`, optional): The minimum and maximum values for each parameters.
        msd_error (:py:attr:`array_like`): Normal uncertainty in the mean squared displacement data. Not necessary if :py:attr:`msd` is :py:attr:`list` of :py:class:`uravu.distribution.Distribution` objects. 
    """
    def __init__(self, delta_t, msd, bounds, msd_errors=None):
        super().__init__(utils.straight_line, delta_t, msd, bounds, ordinate_error=msd_errors)

    @property
    def diffusion_coefficient(self):
        """
        Get the diffusion coefficient found as the gradient divide by 6 (twice the number of dimensions). 

        Returns:
            (:py:class:`uncertainties.core.Variable` or :py:class:`uravu.distribution.Distribution`): The diffusion coefficient in the input units.
        """
        return Distribution(self.variables[0].samples / 6, r'$D$')
        