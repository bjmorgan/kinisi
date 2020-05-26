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
from uravu.axis import Axis
from uravu import utils


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
        self.delta_t = delta_t[::sub_sample_dt]
        self.max_obs = self.displacements[0].shape[1]
        self.distributions = []
        self.dt = np.array([])
        self.iterator = _iterator(progress, range(len(self.displacements)))


class MSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the squared displacements. 
    This resampling method is applied until the MSD distribution is normal (or the `max_resamples` has been reached) and therefore may be described with a median and confidence interval.

    Attributes:
        msd_observed (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations.
        msd_sampled (:py:attr:`array_like`): The population mean-squared displacements, found from the bootstrap resampling of the observations.
        msd_sampled_err (:py:attr:`array_like`): The two-dimensional uncertainties, at the given confidence interval, found from the bootstrap resampling of the observations.

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values.
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point.
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`.
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation).
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`.
        bootstrap_multiplier (:py:attr:`int`, optional): The factor by which the number of bootstrap samples should be multiplied. The default is :py:attr:`1`, which is the maximum number of truely independent samples in a given timestep. This can be increase, however it is importance to note that when greater than 1 the sampling is no longer independent.
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`.
    """
    def __init__(self, delta_t, disp_3d, n_resamples=1000, sub_sample_dt=1, confidence_interval=None, max_resamples=10000, bootstrap_multiplier=1, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, confidence_interval, progress)
        self.msd_observed = np.array([])
        for i in self.iterator:
            d_squared = np.sum(self.displacements[i] ** 2, axis=2)
            n_samples_msd = _n_samples(self.displacements[i].shape, self.max_obs, bootstrap_multiplier)
            if n_samples_msd <= 1:
                continue
            self.msd_observed = np.append(self.msd_observed, np.mean(d_squared.flatten()))
            distro = _sample_until_normal(d_squared, n_samples_msd, n_resamples, max_resamples, self.confidence_interval)
            self.dt = np.append(self.dt, delta_t[i])
            self.distributions.append(distro)
        ax = Axis(self.distributions)
        self.msd_sampled = ax.n
        self.msd_sampled_err = ax.s


class MSCDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the squared charge displacements. 
    This resampling method is applied until the MSCD distribution is normal (or the `max_resamples` has been reached) and therefore may be described with a median and confidence interval.

    Attributes:
        msd_observed (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations.
        msd_sampled (:py:attr:`array_like`): The population mean-squared displacements, found from the bootstrap resampling of the observations.
        msd_sampled_err (:py:attr:`array_like`): The two-dimensional uncertainties, at the given confidence interval, found from the bootstrap resampling of the observations.

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values.
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point.
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`.
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation).
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`.
        bootstrap_multiplier (:py:attr:`int`, optional): The factor by which the number of bootstrap samples should be multiplied. The default is :py:attr:`1`, which is the maximum number of truely independent samples in a given timestep. This can be increase, however it is importance to note that when greater than 1 the sampling is no longer independent.
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`.
    """
    def __init__(self, delta_t, disp_3d, n_resamples=1000, sub_sample_dt=1, confidence_interval=None, max_resamples=10000, bootstrap_multiplier=1, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, confidence_interval, progress)
        self.msd_observed = np.array([])
        for i in self.iterator:
            sq_com_motion = np.sum(self.displacements[i], axis=0) ** 2
            sq_chg_disp = np.sum(sq_com_motion, axis=1)
            n_samples_mscd = _n_samples((1, self.displacements[i].shape[1]), self.max_obs, bootstrap_multiplier)
            if n_samples_mscd <= 1:
                continue
            self.msd_observed = np.append(self.msd_observed, np.mean(sq_chg_disp.flatten())  / self.displacements[i].shape[0])
            distro = _sample_until_normal(sq_chg_disp, n_samples_mscd, n_resamples, max_resamples, self.confidence_interval) 
            self.dt = np.append(self.dt, self.delta_t[i])
            self.distributions.append(Distribution(distro.samples / self.displacements[i].shape[0]))
        ax = Axis(self.distributions)
        self.msd_sampled = ax.n
        self.msd_sampled_err = ax.s


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
        
