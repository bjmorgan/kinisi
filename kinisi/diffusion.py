"""
The modules is focused on tools for the evaluation of the mean squared displacement and resulting
diffusion coefficient from a material.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import warnings
from typing import List, Tuple, Union
import numpy as np
from scipy.stats import multivariate_normal, normaltest, linregress
from scipy.linalg import pinvh
from scipy.optimize import minimize
import scipy.constants as const
import tqdm
from uravu.distribution import Distribution
from sklearn.utils import resample
from emcee import EnsembleSampler 
from kinisi.matrix import find_nearest_positive_definite


class Bootstrap:
    """
    The top-level class for bootstrapping.

    :param delta_t: An array of the timestep values.
    :param disp_3d: A list of arrays, where each array has the axes [atom, displacement observation, dimension].
        There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as
        the number of observations is not necessary the same at each data point.
    :param sub_sample_dt: The frequency in observations to be sampled. Optional, default is :py:attr:`1` (every
        observation).
    :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`.
    """
    def __init__(self, delta_t: np.ndarray, disp_3d: List[np.ndarray], sub_sample_dt: int = 1, progress: bool = True):
        self._displacements = disp_3d[::sub_sample_dt]
        self._delta_t = np.array(delta_t[::sub_sample_dt])
        self._max_obs = self._displacements[0].shape[1]
        self._distributions = []
        self._dt = np.array([])
        self._iterator = self.iterator(progress, range(len(self._displacements)))
        self._n = np.array([])
        self._s = np.array([])
        self._v = np.array([])
        self._n_i = np.array([], dtype=int)
        self._ngp = np.array([])
        self._euclidian_displacements = []
        self._diffusion_coefficient = None
        self._jump_diffusion_coeffiecient = None
        self._sigma = None
        self._intercept = None
        self._covariance_matrix = None

    @property
    def dt(self) -> np.ndarray:
        """
        :return: Timestep values that were resampled.
        """
        return self._dt

    @property
    def n(self) -> np.ndarray:
        """
        :return: The mean MSD/TMSD/MSCD, as determined from the bootstrap resampling process, in units Å:sup:`2`.
        """
        return self._n

    @property
    def s(self) -> np.ndarray:
        """
        :return: The MSD/TMSD/MSCD standard deviation, as determined from the bootstrap resampling process, in
            units Å:sup:`2`.
        """
        return self._s

    @property
    def v(self) -> np.ndarray:
        """
        :return: The MSD/TMSD/MSCD variance as determined from the bootstrap resampling process, in units Å:sup:`4`.
        """
        return self._v

    @property
    def euclidian_displacements(self) -> List[Distribution]:
        """
        :return: Displacements between particles at each dt.
        """
        return self._euclidian_displacements

    @property
    def ngp(self) -> np.ndarray:
        """
        :return: Non-Gaussian parameter as a function of :py:attr:`dt`.
        """
        return self._ngp

    @property
    def n_i(self) -> np.ndarray:
        """
        :return: The number of independent trajectories as a function of :py:attr:`dt`.
        """
        return self._n_i

    @property
    def intercept(self) -> Union[Distribution, None]:
        """
        :return: The estimated intercept. Note that if :py:attr:`fit_intercept` is :py:attr:`False` is the
            relavent method call, then this is :py:attr:`None`
        """
        return self._intercept

    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        :return: The covariance matrix for the trajectories.
        """
        return self._covariance_matrix

    @staticmethod
    def iterator(progress: bool, loop: Union[list, range]) -> Union[tqdm.std.tqdm, range]:
        """
        Get the iteration object, using :py:mod:`tqdm` as appropriate.

        :param progress: Should :py:mod:`tqdm` be used to give a progress bar.
        :param loop: The object that should be looped over.

        :return: Iterator object.
        """
        if progress:
            return tqdm.tqdm(loop, desc='Bootstrapping Displacements')
        return loop

    @staticmethod
    def sample_until_normal(array: np.ndarray,
                            n_samples: int,
                            n_resamples: int,
                            max_resamples: int,
                            alpha: float = 1e-3,
                            random_state: np.random.mtrand.RandomState = None) -> Distribution:
        """
        Resample from the distribution until a normal distribution is obtained or a maximum is reached.

        Args:
        :param array: The array to sample from.
        :param n_samples: Number of samples.
        :param r_resamples: Number of resamples to perform initially.
        :param max_resamples: The maximum number of resamples to perform.
        :param alpha: Level that p-value should be below in :py:func:`scipy.stats.normaltest` for the distribution
            to be normal. Optional, default is :py:attr:`1e-3`.
        :param random_state: A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
            default is :py:attr:`None`.

        :return: The resampled distribution.
        """
        values = _bootstrap(array.flatten(), n_samples, n_resamples, random_state)
        p_value = normaltest(values)[1]
        while (p_value < alpha and len(values) < max_resamples):
            values += _bootstrap(array.flatten(), n_samples, 100, random_state)
        if len(values) >= max_resamples:
            warnings.warn("The maximum number of resamples has been reached, and the distribution is not yet normal.")
        return Distribution(values)

    @staticmethod
    def n_samples(disp_shape: Tuple[float, float], max_obs: int) -> int:
        """
        Calculate the maximum number of independent observations.

        :param disp_shape:: The shape of the displacements array.
        :param max_obs: The maximum number of observations for the trajectory.

        :return: Maximum number of independent observations.
        """
        n_obs = disp_shape[1]
        n_atoms = disp_shape[0]
        dt_int = max_obs - n_obs + 1
        return int(max_obs / dt_int * n_atoms)

    @staticmethod
    def ngp_calculation(d_squared: np.ndarray) -> float:
        """
        Determine the non-Gaussian parameter, from S. Song et al, "Transport dynamics of complex fluids"
        (2019): 10.1073/pnas.1900239116

        :param d_squared: Squared displacement values.

        :return: Value of non-Gaussian parameter.
        """
        top = np.mean(d_squared * d_squared) * 3
        bottom = np.square(np.mean(d_squared)) * 5
        return top / bottom - 1

    def bootstrap_GLS(self,
                      use_ngp: bool = False,
                      dt_skip: float = 0,
                      fit_intercept: bool = True,
                      n_samples: int = 1000,
                      n_walkers: int = 32, 
                      n_burn: int = 500,
                      progress: bool = True,
                      rtol: float = None,
                      random_state: np.random.mtrand.RandomState = None):
        """
        Use the covariance matrix estimated from the resampled values to estimate the gradient and intercept
        using a generalised least squares approach.

        :param use_ngp: Should the ngp max be used as the starting point for the diffusion fitting. Optional,
            default is :py:attr:`False`.
        :param dt_skip: Values of :py:attr:`dt` that should be skipped, i.e. where the atoms are not diffusing.
            Note that if :py:attr:`use_ngp` is :py:attr:`True` this will be ignored. Optional, defaults
            to :py:attr:`0`.
        :param fit_intercept: Should the intercept of the diffusion relationship be fit. Optional, default
            is :py:attr:`True`.
        :param n_samples: Number of samples of the Gaussian process to perform. Optional, default is :py:attr:`1000`.
        :param n_walkers: Number of MCMC walkers to use. Optional, default is :py:attr:`32`.
        :param n_burn: Number of burn in samples (these allow the sampling to settle). Optional, default
            is :py:attr:`500`.
        :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`.
        :param rtol: The relative threshold term for the covariance matrix inversion. If you obtain a very unusual
            value for the diffusion coefficient, it is recommended to increase this value (ideally iteratively). 
            Option, default is :code:`N * eps`, where :code:`eps` is the machine precision value of the covariance 
            matrix content.
        :param random_state: A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
            default is :py:attr:`None`.
        """
        max_ngp = np.argwhere(self._dt > dt_skip)[0][0]
        if use_ngp:
            max_ngp = np.argmax(self._ngp)

        self._covariance_matrix = self.populate_covariance_matrix(self._v, self._n_i)[max_ngp:, max_ngp:]
        self._covariance_matrix = pinvh(pinvh(self._covariance_matrix, rtol=rtol))
        self._covariance_matrix = find_nearest_positive_definite(self._covariance_matrix)

        mv = multivariate_normal(self._n[max_ngp:], self._covariance_matrix, allow_singular=True, seed=random_state)

        def log_likelihood(theta: np.ndarray) -> float:
            """
            Get the log likelihood for multivariate normal distribution.
            :param theta: Value of the gradient and intercept of the straight line.
            :return: Log-likelihood value.
            """
            if theta[0] < 0:
                return -np.inf
            model = _straight_line(self._dt[max_ngp:], *theta)
            logl = mv.logpdf(model)
            return logl

        ols = linregress(self._dt[max_ngp:], self._n[max_ngp:])
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
        self.flatchain = sampler.get_chain(flat=True, discard=n_burn)

        self.gradient = Distribution(self.flatchain[:, 0])
        self._intercept = None
        if fit_intercept:
            self._intercept = Distribution(self.flatchain[:, 1])

    @staticmethod
    def populate_covariance_matrix(variances: np.ndarray, n_samples: np.ndarray) -> np.ndarray:
        """
        Populate the covariance matrix for the generalised least squares methodology.

        :param variances: The variances for each timestep
        :param n_samples: Number of independent trajectories for each timestep

        :return: An estimated covariance matrix for the system
        """
        covariance_matrix = np.zeros((variances.size, variances.size))
        for i in range(0, variances.size):
            for j in range(i, variances.size):
                value = n_samples[i] / n_samples[j] * variances[i]
                covariance_matrix[i, j] = value
                covariance_matrix[j, i] = np.copy(covariance_matrix[i, j])
        return covariance_matrix


class MSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the mean
    squared displacements.

    :param delta_t: An array of the timestep values, units of ps
    :param disp_3d: A list of arrays, where each array has the axes
        :py:code:`[atom, displacement observation, dimension]`. There is one array in the list for each
        delta_t value. Note: it is necessary to use a list of arrays as the number of observations is
        not necessary the same at each data point.
    :param sub_sample_dt: The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
    :param n_resamples: The initial number of resamples to be performed. Default is :py:attr:`1000`
    :param max_resamples: The max number of resamples to be performed by the distribution is assumed to be normal.
        This is present to allow user control over the time taken for the resampling to occur. Default
        is :py:attr:`100000`
    :param alpha: Value that p-value for the normal test must be greater than to accept. Default is :py:attr:`1e-3`
    :param random_state : A :py:attr:`RandomState` object to be used to ensure reproducibility. Default
        is :py:attr:`None`
    :param progress: Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 sub_sample_dt: int = 1,
                 n_resamples: int = 1000,
                 max_resamples: int = 10000,
                 alpha: float = 1e-3,
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, progress)
        for i in self._iterator:
            d_squared = np.sum(self._displacements[i]**2, axis=2)
            n_samples_current = self.n_samples(self._displacements[i].shape, self._max_obs)
            if n_samples_current <= 1:
                continue
            self._n_i = np.append(self._n_i, n_samples_current)
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            distro = self.sample_until_normal(d_squared, self._n_i[i], n_resamples, max_resamples, alpha, random_state)
            self._distributions.append(distro)
            self._n = np.append(self._n, distro.n)
            self._s = np.append(self._s, np.std(distro.samples, ddof=1))
            self._v = np.append(self._v, np.var(distro.samples, ddof=1))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared))
            self._dt = np.append(self._dt, self._delta_t[i])

    def diffusion(self, **kwargs):
        """
        Use the bootstrap-GLS method to determine the diffusivity for the system. Keyword arguments will be
        passed of the :py:func:`bootstrap_GLS` method.
        """
        self.bootstrap_GLS(**kwargs)
        self._diffusion_coefficient = Distribution(self.gradient.samples / (2e4 * self._displacements[0].shape[-1]))

    @property
    def D(self) -> Union[Distribution, None]:
        """
        An alias for the diffusion coefficient Distribution.

        :return: Diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`.
        """
        return self._diffusion_coefficient


class TMSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the total
    mean squared displacements.

    :param delta_t: An array of the timestep values.
    :param disp_3d: A list of arrays, where each array has the axes
        :py:code:`[atom, displacement observation, dimension]`. There is one array in the list for each
        delta_t value. Note: it is necessary to use a list of arrays as the number of observations is
        not necessary the same at each data point.
    :param sub_sample_dt: The frequency in observations to be sampled. Optional, default
        is :py:attr:`1` (every observation)
    :param n_resamples: The initial number of resamples to be performed. Optional, default
        is :py:attr:`1000`
    :param max_resamples: The max number of resamples to be performed by the distribution is assumed to be
        normal. This is present to allow user control over the time taken for the resampling to occur.
        Optional, default is :py:attr:`100000`
    :param alpha: Value that p-value for the normal test must be greater than to accept. Optional, default
        is :py:attr:`1e-3`
    :param random_state : A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
        default is :py:attr:`None`
    :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`
    """
    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 sub_sample_dt: int = 1,
                 n_resamples: int = 1000,
                 max_resamples: int = 10000,
                 alpha: float = 1e-3,
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, progress)
        for i in self._iterator:
            d_squared = np.sum(self._displacements[i]**2, axis=2)
            coll_motion = np.sum(np.sum(self._displacements[i], axis=0)**2, axis=-1)
            n_samples_current = self.n_samples((1, self._displacements[i].shape[1]), self._max_obs)
            if n_samples_current <= 1:
                continue
            self._n_i = np.append(self._n_i, n_samples_current)
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            distro = self.sample_until_normal(coll_motion, self._n_i[i], n_resamples, max_resamples, alpha,
                                              random_state)
            self._distributions.append(distro)
            self._n = np.append(self._n, distro.n)
            self._s = np.append(self._s, np.std(distro.samples, ddof=1))
            self._v = np.append(self._v, np.var(distro.samples, ddof=1))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared.flatten()))
            self._dt = np.append(self._dt, self._delta_t[i])

    def jump_diffusion(self, **kwargs):
        """
        Use the bootstrap-GLS method to determine the jump diffusivity for the system. Keyword arguments
        will be passed of the :py:func:`bootstrap_GLS` method.
        """
        self.bootstrap_GLS(**kwargs)
        self._jump_diffusion_coefficient = Distribution(
            self.gradient.samples / (2e4 * self._displacements[0].shape[-1] * self._displacements[0].shape[0]))

    @property
    def D_J(self) -> Union[Distribution, None]:
        """
        Alias for the jump diffusion coefficient Distribution.

        :return: Jump diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`.
        """
        return self._jump_diffusion_coefficient


class MSCDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the mean
    squared charge displacements.

    :param delta_t: An array of the timestep values.
    :param disp_3d: A list of arrays, where each array has the axes
        :py:code:`[atom, displacement observation, dimension]`. There is one array in the list for each
        delta_t value. Note: it is necessary to use a list of arrays as the number of observations is
        not necessary the same at each data point.
    :param ionic_charge: The charge on the mobile ions, either an array with a value for each ion or a scalar
        if all values are the same.
    :param sub_sample_dt: The frequency in observations to be sampled. Optional, default is :py:attr:`1`
        (every observation).
    :param n_resamples: The initial number of resamples to be performed. Optional, default is :py:attr:`1000`.
    :param max_resamples: The max number of resamples to be performed by the distribution is assumed to be normal.
        This is present to allow user control over the time taken for the resampling to occur. Optional, default
        is :py:attr:`100000`.
    :param alpha: Value that p-value for the normal test must be greater than to accept. Optional, default
        is :py:attr:`1e-3`.
    :param random_state: A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
        default is :py:attr:`None`.
    :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`.
    """
    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 ionic_charge: Union[np.ndarray, int],
                 sub_sample_dt: int = 1,
                 n_resamples: int = 1000,
                 max_resamples: int = 10000,
                 alpha: float = 1e-3,
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, progress)
        try:
            _ = len(ionic_charge)
        except TypeError:
            ionic_charge = np.ones(self._displacements[0].shape[0]) * ionic_charge
        for i in self._iterator:
            d_squared = np.sum(self._displacements[i]**2, axis=2)
            sq_chg_motion = np.sum(np.sum((ionic_charge * self._displacements[i].T).T, axis=0)**2, axis=-1)
            n_samples_current = self.n_samples((1, self._displacements[i].shape[1]), self._max_obs)
            if n_samples_current <= 1:
                continue
            self._n_i = np.append(self._n_i, n_samples_current)
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            distro = self.sample_until_normal(sq_chg_motion, self._n_i[i], n_resamples, max_resamples, alpha,
                                              random_state)
            self._distributions.append(distro)
            self._n = np.append(self._n, distro.n)
            self._s = np.append(self._s, np.std(distro.samples, ddof=1))
            self._v = np.append(self._v, np.var(distro.samples, ddof=1))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared.flatten()))
            self._dt = np.append(self._dt, self._delta_t[i])

    def conductivity(self, temperature: float, volume: float, **kwargs):
        """
        Use the bootstrap-GLS method to determine the ionic conductivity for the system, in units of mScm:sup:`-1`.
        Keyword arguments will be passed of the :py:func:`bootstrap_GLS` method.

        :param temperature: System temperature, in Kelvin.
        :param volume: System volume, in Å^{3}.
        """
        self.bootstrap_GLS(**kwargs)
        volume = volume * 1e-24  # cm^3
        D = self.gradient.samples / (2e4 * self._displacements[0].shape[-1])  # cm^2s^-1
        conversion = 1000 / (volume * const.N_A) * (const.N_A * const.e)**2 / (const.R * temperature)
        self._sigma = Distribution(D * conversion)

    @property
    def sigma(self) -> Union[Distribution, None]:
        """
        :return: The estimated conductivity, based on the generalised least squares approach, with
            units mScm:sup:`-1`.
        """
        return self._sigma


def _bootstrap(array: np.ndarray, n_samples: int, n_resamples: int, random_state: np.random.mtrand.RandomState = None):
    """
    Perform a set of resamples.

    :param array: The array to sample from.
    :param n_samples: Number of samples.
    :param n_resamples: Number of resamples to perform initially.
    :param random_state: A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
        default is :py:attr:`None`

    :return: Resampled values from the array
    """
    return [
        np.mean(resample(array.flatten(), n_samples=n_samples, random_state=random_state)) for j in range(n_resamples)
    ]


def _straight_line(abscissa: np.ndarray, gradient: float, intercept: float = 0.0) -> np.ndarray:
    """
    A one dimensional straight line function.

    :param abscissa: The abscissa data.
    :param gradient: The slope of the line.
    :param intercept: The y-intercept of the line. Optional, default is :py:attr:`0.0`.

    :return: The resulting ordinate.
    """
    return gradient * abscissa + intercept
