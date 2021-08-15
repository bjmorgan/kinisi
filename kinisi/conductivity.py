"""
The modules is focused on tools for the evaluation of the mean squared displacement and resulting diffusion coefficient from a material.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import warnings
import numpy as np
from scipy.stats import linregress, multivariate_normal, normaltest
from scipy.optimize import minimize
from tqdm import tqdm
from uravu.distribution import Distribution
from sklearn.utils import resample
from emcee import EnsembleSampler
from kinisi.matrix import find_nearest_positive_definite
from kinisi import diffusion


class MSCDBootstrap(diffusion.Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the squared displacements.
    This resampling method is applied until the final MSD distribution is normal (or :py:code:`max_resamples` has been reached).

    Attributes:
        msd (:py:attr:`array_like`): The sample mean-squared displacements, found from the arithmetic average of the observations
        msd_std (:py:attr:`array_like`): The standard deviation in the mean-squared displacement, found from the bootstrap resampling of the trajectories
        correlation_matrix (:py:attr:`array_like`): The estimated correlation matrix for the mean-squared displacements
        ngp (:py:attr:`array_like`): Non-Gaussian parameter as a function of dt
        euclidian_displacements (:py:attr:`list` of :py:class:`uravu.distribution.Distribution`): Displacements between particles at each dt
        covariance_matrix (:py:attr:`array_like`): The covariance matrix for the trajectories

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`
        random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:code:`RandomState` object to be used to ensure reproducibility. Default is :py:code:`None`
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self, delta_t, disp_3d, sub_sample_dt=1, n_resamples=1000, max_resamples=10000, alpha=1e-3, confidence_interval=[2.5, 97.5], random_state=None, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, progress)
        self.msd_var = np.array([])
        self.n_samples_msd = np.array([], dtype=int)
        self.ngp = np.array([])
        self.euclidian_displacements = []
        for i in self.iterator:
            d_squared = np.sum(self.displacements[i] ** 2, axis=2)
            sq_com_motion = np.sum(self.displacements[i], axis=0) ** 2
            sq_chg_disp = np.sum(sq_com_motion, axis=1)
            n_samples_current = diffusion.MSDBootstrap.n_samples((1, self.displacements[i].shape[1]), self.max_obs)
            if n_samples_current <= 1:
                continue
            self.n_samples_msd = np.append(self.n_samples_msd, n_samples_current)
            self.euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            distro = diffusion.MSDBootstrap.sample_until_normal(d_squared, self.n_samples_msd[i], n_resamples, max_resamples, alpha, confidence_interval, random_state)
            self.distributions.append(Distribution(distro.samples / self.displacements[i].shape[0]))
            self.msd = np.append(self.msd, distro.n)
            self.msd_std = np.append(self.msd_std, np.std(distro.samples, ddof=1))
            self.msd_var = np.append(self.msd_var, np.var(distro.samples, ddof=1))
            self.ngp = np.append(self.ngp, diffusion.MSDBootstrap.ngp_calculation(d_squared))
            self.dt = np.append(self.dt, self.delta_t[i])

    def conductivity(self, use_ngp=False, fit_intercept=True, n_walkers=32, n_samples=1000, random_state=None, progress=True):
        """
        Use the covariance matrix estimated from the resampled values to estimate the diffusion coefficient and intercept using a generalised least squares approach.
            
        Args:
            use_ngp (:py:attr:`bool`, optional): Should the ngp max be used as the starting point for the diffusion fitting. Default is :py:attr:`False`
            fit_intercept (:py:attr:`bool`, optional): Should the intercept of the diffusion relationship be fit. Default is :py:attr:`True`.
            n_walkers (:py:attr:`int`, optional): Number of MCMC walkers to use. Default is :py:attr:`32`.
            n_samples (:py:attr:`int`, optional): Number of MCMC samples to perform. Default is :py:attr:`1000`.
            random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:code:`RandomState` object to be used to ensure reproducibility. Default is :py:code:`None`
            progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
        """
        max_ngp = 0 
        if use_ngp:
            max_ngp = np.argmax(self.ngp)
        self.covariance_matrix = diffusion.MSDBootstrap.populate_covariance_matrix(self.msd_var, self.n_samples_msd)[max_ngp:, max_ngp:]
        self.covariance_matrix = find_nearest_positive_definite(self.covariance_matrix)

        mv = multivariate_normal(self.msd[max_ngp:], self.covariance_matrix, allow_singular=True, seed=random_state)

        def log_likelihood(theta):
            """
            Get the log likelihood for multivariate normal distribution.

            Args:
                theta (:py:attr:`array_like`): Value of the gradient and intercept of the straight line.

            Returns:
                (:py:attr:`float`): Log-likelihood value.
            """
            model = diffusion._straight_line(self.dt[max_ngp:], *theta)
            logl = mv.logpdf(model)
            return logl
        ols = linregress(self.dt[max_ngp:], self.msd[max_ngp:])

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
        # Waiting on https://github.com/dfm/emcee/pull/376
        # if random_state is not None:
        #     pos = max_likelihood + max_likelihood * 1e-3 * random_state.randn(n_walkers, max_likelihood.size)
        #     sampler._random = random_state
        sampler.run_mcmc(pos, n_samples, progress=progress, progress_kwargs={'desc': "Likelihood Sampling"})
        flatchain = sampler.get_chain(flat=True)
        self.conductivity = Distribution(flatchain[:, 0] / 6)
        self.intercept = None
        if fit_intercept:
            self.intercept = Distribution(flatchain[:, 1])

    @property
    def sigma(self):
        """
        An alias for the diffusion coefficient Distribution.

        Returns:
            (:py:class:`uravu.distribution.Distribution`): Diffusion coefficient
        """
        return self.conductivity

    @property
    def sigma_offset(self):
        """
        An alias for the diffusion coefficient offset Distribution.

        Returns:
            (:py:class:`uravu.distribution.Distribution`): Diffusion coefficient offset
        """
        return self.intercept