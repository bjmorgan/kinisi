"""
The modules is focused on tools for the evaluation of the mean squared displacement and resulting diffusion coefficient from a material.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import warnings
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.linalg import pinv
from sklearn.utils import resample
from tqdm import tqdm
from emcee import EnsembleSampler
from uravu.distribution import Distribution
from uravu.utils import straight_line


class Bootstrap:
    """
    The top-level class for bootstrapping.

    Attributes:
        confidence_interval (:py:attr:`array_like`): The percentile points of the distribution that should be stored
        displacements (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        delta_t (:py:attr:`array_like`): An array of the timestep values
        max_obs (:py:attr:`int`): The maximum number of observations for the trajectory
        dt (:py:attr:`array_like`): Timestep values that were resampled
        distributions (:py:attr:`list` of :py:class:`uravu.distribution.Distribution`): Resampled mean squared distributions
        iterator (:py:attr:`tqdm` or :py:attr:`range`): The iteration object


    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval)
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self, delta_t, disp_3d, sub_sample_dt=1, confidence_interval=None, progress=True):
        if confidence_interval is None:
            self.confidence_interval = [2.5, 97.5]
        self.displacements = disp_3d[::sub_sample_dt]
        self.delta_t = np.array(delta_t[::sub_sample_dt])
        self.max_obs = self.displacements[0].shape[1]
        self.distributions = []
        self.distributions_4 = []
        self.dt = np.array([])
        self.iterator = _iterator(progress, range(len(self.displacements)))
        self.diffusion_coefficient = None
        self.intercept = None

    @property
    def D(self):
        """
        An alias for the diffusion coefficient Distribution.

        Returns:
            (:py:class:`uravu.distribution.Distribution`): Diffusion coefficient
        """
        return self.diffusion_coefficient


class MSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the squared displacements.
    This resampling method is applied until the MSD distribution is normal (or the `max_resamples` has been reached) and therefore may be described with a median and confidence interval.

    Attributes:
        msd_observed (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations
        msd_sampled (:py:attr:`array_like`): The population mean-squared displacements, found from the bootstrap resampling of the observations
        msd_sampled_err (:py:attr:`array_like`): The uncertainties, at the given confidence interval, found from the bootstrap resampling of the observations
        msd_sampled_std (:py:attr:`array_like`): The standard deviation in the mean-squared displacement, found from the bootstrap resampling of the observations
        correlation_matrix (:py:attr:`array_like`): The estimated correlation matrix for the mean-squared displacements
        ngp (:py:attr:`array_like`): Non-Gaussian parameter as a function of dt
        ngp_err (:py:attr:`array_like`): Non-Gaussian parameter uncertainty as a function of dt, if found
        euclidian_displacements (:py:attr:`list` of :py:class:`uravu.distribution.Distribution`): Displacements between particles at each dt
        n_sampled_msd (:py:attr:`array_like`): The number of independent trajectories as a function of dt

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
        confidence_interval (:py:attr:`array_like`, optional): The percentile points of the distribution that should be stored. Default is :py:attr:`[31.73, 68.27]` (a single standard deviation)
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`
        bootstrap_multiplier (:py:attr:`int`, optional): The factor by which the number of bootstrap samples should be multiplied. The default is :py:attr:`1`, which is the maximum number of truely independent samples in a given timestep. This can be increase, however it is importance to note that when greater than 1 the sampling is no longer independent
        ngp_errors (:py:attr:`bool`, optional): Should the non-Gaussian parameter uncertainty be calculated. Default is :py:attr:`True`
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self, delta_t, disp_3d, n_resamples=1000, sub_sample_dt=1, confidence_interval=None,
                 max_resamples=10000, bootstrap_multiplier=1, ngp_errors=False, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, confidence_interval, progress)
        self.msd_observed = np.array([])
        self.msd_sampled = np.array([])
        self.msd_sampled_err = np.array([])
        self.msd_sampled_std = np.array([])
        self.ngp = np.array([])
        self.ngp_err = None
        if ngp_errors:
            self.ngp_err = np.array([])
        self.euclidian_displacements = []
        self.n_samples_msd = np.array([], dtype=int)
        for i in self.iterator:
            d_squared = np.sum(self.displacements[i] ** 2, axis=2)
            self.euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            self.n_samples_msd = np.append(self.n_samples_msd, _n_samples(self.displacements[i].shape, self.max_obs, bootstrap_multiplier))
            if self.n_samples_msd[i] <= 1:
                continue
            self.msd_observed = np.append(self.msd_observed, np.mean(d_squared.flatten()))
            distro = _sample_until_normal(d_squared, self.n_samples_msd[i], n_resamples, max_resamples, self.confidence_interval)
            if ngp_errors:
                distro4 = _sample_until_normal(d_squared * d_squared, self.n_samples_msd[i], n_resamples, max_resamples, self.confidence_interval)
                self.distributions_4.append(distro4)
                top = distro4.samples[np.random.choice(distro4.size, size=1000)] * 3
                bottom = np.square(distro.samples[np.random.choice(distro.size, size=1000)]) * 5
                ngp_d = Distribution(top / bottom - 1, ci_points=self.confidence_interval)
                self.ngp = np.append(self.ngp, ngp_d.n)
                self.ngp_err = np.append(self.ngp_err, distro4.n - distro4.con_int[0])
            else:
                top = np.mean(d_squared.flatten() * d_squared.flatten()) * 3
                bottom = np.square(np.mean(d_squared.flatten())) * 5
                self.ngp = np.append(self.ngp, top / bottom - 1)
            self.dt = np.append(self.dt, self.delta_t[i])
            self.distributions.append(distro)
            self.msd_sampled = np.append(self.msd_sampled, distro.n)
            self.msd_sampled_err = np.append(self.msd_sampled_err, distro.n - distro.con_int[0])
            self.msd_sampled_std = np.append(self.msd_sampled_std, np.std(distro.samples))

    def diffusion(self, fit_intercept=True, use_ngp=True, n_samples=1000, n_walkers=32, progress=False):
        """
        Calculate the diffusion coefficient for the trajectory.

        Args:
            fit_intercept (:py:attr:`bool`, optional): Should the intercept of the diffusion relationship be fit. Default is :py:attr:`True`.
            use_ngp (:py:attr:`bool`, optional): Should the ngp max be used as the starting point for the diffusion fitting. Default is :py:attr:`True`
            n_samples (:py:attr:`int`, optional): Number of likelihood samples to perform. Default is :py:attr:`1000`.
            n_walkers (:py:attr:`int`, optional): Number of likelihood walkers to use. Default is :py:attr:`32`.
            progress (:py:attr:`bool`, optional): Show tqdm progress for likelihood sampling. Default is :py:attr:`False`.
        """
        max_ngp = np.argmax(self.ngp)
        if not use_ngp:
            max_ngp = 0
        self.covariance_matrix = np.zeros((self.dt.size, self.dt.size))
        for i in range(0, self.dt.size):
            for j in range(i, self.dt.size):
                ratio = self.n_samples_msd[i] / self.n_samples_msd[j]
                self.covariance_matrix[i, j] = np.var(self.distributions[i].samples) * ratio
                self.covariance_matrix[j, i] = np.copy(self.covariance_matrix[i, j])
        ln_sigma = np.multiply(*np.linalg.slogdet(self.covariance_matrix[max_ngp:, max_ngp:]))
        inv = pinv(self.covariance_matrix[max_ngp:, max_ngp:], atol=self.covariance_matrix[max_ngp:, max_ngp:].min())
        end = self.msd_sampled[max_ngp:].size * np.log(2. * np.pi)

        def log_likelihood(theta):
            """
            Get the log likelihood for multivariate normal distribution.

            Args:
                theta (:py:attr:`array_like`): Value of the gradient and intercept of the straight line.

            Returns:
                (:py:attr:`float`): Log-likelihood value.
            """
            model = straight_line(self.dt[max_ngp:], *theta)
            difference = np.subtract(model, self.msd_sampled[max_ngp:])
            logl = -0.5 * (ln_sigma + np.matmul(difference.T, np.matmul(inv, difference)) + end)
            return logl
        ols = linregress(self.dt[max_ngp:], self.msd_sampled[max_ngp:])

        def nll(*args):
            """
            General purpose negative log-likelihood.

            Returns:
                (:py:attr:`float`): Negative log-likelihood.
            """
            return -log_likelihood(*args)
        if fit_intercept:
            max_likelihood = minimize(nll, np.array([ols.slope, ols.intercept])).x
        else:
            max_likelihood = minimize(nll, np.array([ols.slope])).x
        pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(n_walkers, max_likelihood.size)
        sampler = EnsembleSampler(*pos.shape, log_likelihood)
        sampler.run_mcmc(pos, n_samples, progress=progress)
        self.diffusion_coefficient = Distribution(sampler.flatchain[:, 0] / 6, ci_points=self.confidence_interval)
        if fit_intercept:
            self.intercept = Distribution(sampler.flatchain[:, 1], ci_points=self.confidence_interval)


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
