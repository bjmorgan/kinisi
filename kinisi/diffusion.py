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
import pandas as pd
from scipy.stats import uniform, norm, multivariate_normal
from sklearn.utils import resample
from tqdm import tqdm
from statsmodels.stats.moment_helpers import corr2cov
from uravu.distribution import Distribution
from uravu.axis import Axis


class Bootstrap:
    """
    The top-level class for bootstrapping.

    Attributes:
        confidence_interval (:py:attr:`array_like`): The percentile points of the distribution that should be stored.
        displacements (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point.
        delta_t (:py:attr:`array_like`): An array of the timestep values.
        max_obs (:py:attr:`int`): The maximum number of observations for the trajectory.
        dt (:py:attr:`array_like`): Timestep values that were resampled.
        distributions ():py:attr:`list` of :py:class:`uravu.distribution.Distribution`): Resampled mean squared distributions.
        iterator (:py:attr:`tqdm` or :py:attr:`range`): The iteration object.


    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values.
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point.
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation).
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`.
    """
    def __init__(self, delta_t, disp_3d, sub_sample_dt=1, confidence_interval=None, progress=True):
        if confidence_interval is None:
            self.confidence_interval = [2.5, 97.5]
        self.displacements = disp_3d[::sub_sample_dt]
        self.delta_t = np.array(delta_t[::sub_sample_dt])
        self.max_obs = self.displacements[0].shape[1]
        self.distributions = []
        self.dt = np.array([])
        self.iterator = _iterator(progress, range(len(self.displacements)))
        self.diffusion_coefficient = None
        self.intercept = None

    @property
    def D(self):
        """
        An alias for the diffusion coefficient Distribution.
        """
        return self.diffusion_coefficient


class MSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the squared displacements.
    This resampling method is applied until the MSD distribution is normal (or the `max_resamples` has been reached) and therefore may be described with a median and confidence interval.

    Attributes:
        msd_observed (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations.
        msd_sampled (:py:attr:`array_like`): The population mean-squared displacements, found from the bootstrap resampling of the observations.
        msd_sampled_err (:py:attr:`array_like`): The uncertainties, at the given confidence interval, found from the bootstrap resampling of the observations.
        msd_sampled_std (:py:attr:`array_like`): The standard deviation in the mean-squared displacement, found from the bootstrap resampling of the observations.
        correlation_matrix (:py:attr:`array_like`): The estimated correlation matrix for the mean-squared displacements.

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values.
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point.
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`.
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation).
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[31.73, 68.27]` (a single standard deviation).
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`.
        bootstrap_multiplier (:py:attr:`int`, optional): The factor by which the number of bootstrap samples should be multiplied. The default is :py:attr:`1`, which is the maximum number of truely independent samples in a given timestep. This can be increase, however it is importance to note that when greater than 1 the sampling is no longer independent.
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`.
    """
    def __init__(self, delta_t, disp_3d, n_resamples=1000, sub_sample_dt=1, confidence_interval=None, max_resamples=10000, bootstrap_multiplier=1, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, confidence_interval, progress)
        self.msd_observed = np.array([])
        self.msd_sampled = np.array([])
        self.msd_sampled_err = np.array([])
        self.msd_sampled_std = np.array([])
        samples = np.zeros((self.displacements[0].shape[0], len(self.displacements)))
        for i in self.iterator:
            d_squared = np.sum(self.displacements[i] ** 2, axis=2)
            samples[:, i] = d_squared.mean(axis=1).flatten()
            n_samples_msd = _n_samples(self.displacements[i].shape, self.max_obs, bootstrap_multiplier)
            if n_samples_msd <= 1:
                continue
            self.msd_observed = np.append(self.msd_observed, np.mean(d_squared.flatten()))
            distro = _sample_until_normal(d_squared, n_samples_msd, n_resamples, max_resamples, self.confidence_interval)
            self.dt = np.append(self.dt, self.delta_t[i])
            self.distributions.append(distro)
            self.msd_sampled = np.append(self.msd_sampled, distro.n)
            self.msd_sampled_err = np.append(self.msd_sampled_err, distro.n - distro.con_int[0])
            self.msd_sampled_std = np.append(self.msd_sampled_std, np.std(distro.samples))
        self.correlation_matrix = np.array(pd.DataFrame(samples).corr())

    def diffusion(self, n_samples=10000, fit_intercept=True):
        """
        Calculate the diffusion coefficient for the trajectory.

        Args:
            n_samples (:py:attr:`int`, optional): The number of samples in the random generator. Default is :py:attr:`10000`.
            fit_intercept (:py:attr:`bool`, optional): Should the intercept of the diffusion relationship be fit. Default is :py:attr:`True`.
        """
        single_msd = multivariate_normal(self.msd_sampled, corr2cov(self.correlation_matrix, self.msd_sampled_std), allow_singular=True)
        single_msd_samples = single_msd.rvs(n_samples)
        A = np.array([self.dt]).T
        if fit_intercept:
            A = np.array([np.ones(self.dt.size), self.dt]).T
        Y = single_msd_samples.T
        straight_line = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), Y)
        if fit_intercept:
            intercept, gradient = straight_line
            self.diffusion_coefficient = Distribution(gradient / 6, ci_points=self.confidence_interval)
            self.intercept = Distribution(intercept, ci_points=self.confidence_interval)
        else:
            self.diffusion_coefficient = Distribution(straight_line[0] / 6, ci_points=self.confidence_interval)


def _n_samples(disp_shape, max_obs, bootstrap_multiplier):
    """
    Calculate the maximum number of independent observations.

    Args:
        disp_shape (:py:attr:`tuple`): The shape of the displacements array.
        max_obs (:py:attr:`int`): The maximum number of observations for the trajectory.
        bootstrap_multiplier (:py:attr:`int`, optional): The factor by which the number of bootstrap samples should be multiplied. The default is :py:attr:`1`, which is the maximum number of truely independent samples in a given timestep. This can be increase, however it is importance to note that when greater than 1 the sampling is no longer independent.

    Returns:
        :py:attr:`int`: Maximum number of independent observations.
    """
    n_obs = disp_shape[1]
    n_atoms = disp_shape[0]
    dt_int = max_obs - n_obs + 1
    # approximate number of "non-overlapping" observations, allowing
    # for partial overlap
    return int(max_obs / dt_int * n_atoms) * bootstrap_multiplier


def _iterator(progress, loop):
    """
    Get the iteration object, using :py:mod:`tqdm` as appropriate.

    Args:
        progress (:py:attr:`bool`): Should :py:mod:`tqdm` be used to give a progress bar.
        loop (:py:attr:`list` or :py:attr:`range`): The object that should be looped over.

    Returns:
        :py:attr:`tqdm` or :py:attr:`range`: Iterator object.
    """
    if progress:
        return tqdm(loop, desc='Bootstrapping Displacements')
    else:
        return loop


def _sample_until_normal(array, n_samples, n_resamples, max_resamples, confidence_interval):
    """
    Resample from the distribution until a normal distribution is obtained or a maximum is reached.

    Args:
        array (:py:attr:`array_like`): The array to sample from.
        n_samples (:py:attr:`int`): Number of samples.
        r_resamples (:py:attr:`int`): Number of resamples to perform initially.
        max_resamples (:py:attr:`int`): The maximum number of resamples to perform.
        confidence_interval (:py:attr:`array_like`): The percentile points of the distribution that should be stored.

    Returns:
        :py:class:`uravu.distribution.Distribution`: The resampled distribution.
    """
    distro = Distribution(_bootstrap(array.flatten(), n_samples, n_resamples), ci_points=confidence_interval)
    while (not distro.normal) and distro.size < max_resamples:
        distro.add_samples(_bootstrap(array.flatten(), n_samples, 100))
    if distro.size >= max_resamples:
        warnings.warn("The maximum number of resamples has been reached, and the distribution is not yet normal.")
    return distro


def _bootstrap(array, n_samples, n_resamples):
    """
    Perform a set of resamples.

    Args:
        array (:py:attr:`array_like`): The array to sample from.
        n_samples (:py:attr:`int`): Number of samples.
        r_resamples (:py:attr:`int`): Number of resamples to perform initially.

    Returns:
        :py:attr:`array_like`: Resampled values from the array.
    """
    return [np.mean(resample(array.flatten(), n_samples=n_samples)) for j in range(n_resamples)]
