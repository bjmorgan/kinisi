"""
Calculate the diffusion coefficient.
"""

# Copyright (c) kinisi developers. 
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
from statsmodels.stats.correlation_tools import cov_nearest
from scipy.linalg import pinvh
from scipy.stats import linregress
from scipy.optimize import minimize
from emcee import EnsembleSampler


class Diffusion:
    """
    Class to calculate the diffusion coefficient.
    """
    def __init__(self, msd: sc.DataArray):
        self.msd = msd
    
    @property
    def covariance_matrix(self) -> sc.Variable:
        return self._covariance_matrix

    def bayesian_regression(self,
                            start_dt: sc.Variable,
                            cond_max: float = 1e16,
                            fit_intercept: bool = True,
                            n_samples: int = 1000,
                            n_walkers: int = 32,
                            n_burn: int = 500,
                            n_thin: int = 10,
                            progress: bool = True,
                            random_state: np.random.mtrand.RandomState = None):
        if random_state is not None: 
            np.random.seed(random_state.get_state()[1][1])
            
        self._start_dt = start_dt
        self._cond_max = cond_max
    
        diff_regime = np.argwhere(self.msd.coords['timestep'] >= self._start_dt)[0][0]
        self._covariance_matrix = self.compute_covariance_matrix()
    
        x_values = self.msd.coords['timestep'][diff_regime:].values
        y_values = self.msd['timestep', diff_regime:].values
        
        _, logdet = np.linalg.slogdet(self._covariance_matrix.values[diff_regime:, diff_regime:])
        logdet += np.log(2 * np.pi) * y_values.size
        inv = pinvh(self._covariance_matrix.values[diff_regime:, diff_regime:])

        def log_likelihood(theta: np.ndarray) -> float:
            """
            Get the log likelihood for multivariate normal distribution.
            :param theta: Value of the gradient and intercept of the straight line.
            :return: Log-likelihood value.
            """
            if theta[0] < 0:
                return -np.inf
            model = _straight_line(x_values, *theta)
            diff = (model - y_values)
            logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
            return logl
    
        ols = linregress(x_values, y_values)
        slope = ols.slope
        intercept = 1e-20
        if slope < 0:
            slope = 1e-20

        def nll(*args) -> float:
            """
            General purpose negative log-likelihood.
            :return: Negative log-likelihood
            """
            return -log_likelihood(*args)

        if fit_intercept:
            max_likelihood = minimize(nll, np.array([slope, intercept])).x
        else:
            max_likelihood = minimize(nll, np.array([slope])).x

        pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(n_walkers, max_likelihood.size)
        sampler = EnsembleSampler(*pos.shape, log_likelihood)
        # Waiting on https://github.com/dfm/emcee/pull/376
        # if random_state is not None:
        #     pos = max_likelihood + max_likelihood * 1e-3 * random_state.randn(n_walkers, max_likelihood.size)
        #     sampler._random = random_state
        sampler.run_mcmc(pos, n_samples + n_burn, progress=progress, progress_kwargs={'desc': "Likelihood Sampling"})
        flatchain = sampler.get_chain(flat=True, thin=n_thin, discard=n_burn)
    
        self.gradient = sc.array(dims=['samples'], values=flatchain[:, 0], unit=(self.msd.unit / self.msd.coords['timestep'].unit))
        self.intercept = None
        if fit_intercept:
            self.intercept = sc.array(dims=['samples'], values=flatchain[:, 1], unit=self.msd.unit)

    def diffusion(self, start_dt: sc.Variable, **kwargs):
        self.bayesian_regression(start_dt=start_dt, **kwargs)
        self._diffusion_coefficient = sc.to_unit(self.gradient / (2 * self.msd.coords['dimensionality'].value), 'cm2/s')

    @property
    def D(self) -> sc.Variable:
        return self._diffusion_coefficient

    def compute_covariance_matrix(self) -> sc.Variable:
        cov = np.zeros((self.msd.data.variances.size, self.msd.data.variances.size))
        for i in range(0, self.msd.data.variances.size):
                for j in range(i, self.msd.data.variances.size):
                    ratio = self.msd.coords['n_samples'].values[i] / self.msd.coords['n_samples'].values[j]
                    value = ratio * self.msd.data.variances[i]
                    cov[i, j] = value
                    cov[j, i] = np.copy(cov[i, j])
        return sc.array(dims=['timestep1', 'timestep2'], values=minimum_eigenvalue_method(cov), unit=self.msd.unit**2)


def minimum_eigenvalue_method(cov: np.ndarray) -> sc.Variable:
    ee = np.linalg.eig(cov)
    ev = ee.eigenvalues
    new_ev = np.copy(ev)
    T = ev[0] / 1e16
    new_ev[np.where(new_ev < T)] = T
    new_cov = np.real(ee.eigenvectors @ np.diag(ev) @ ee.eigenvectors.T)
    return cov_nearest(new_cov)


def _straight_line(abscissa: np.ndarray, gradient: float, intercept: float = 0.0) -> np.ndarray:
    """
    A one dimensional straight line function.

    :param abscissa: The abscissa data.
    :param gradient: The slope of the line.
    :param intercept: The y-intercept of the line. Optional, default is :py:attr:`0.0`.

    :return: The resulting ordinate.
    """
    return gradient * abscissa + intercept