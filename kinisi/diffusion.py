"""
The modules is focused on tools for the evaluation of the mean squared displacement and resulting
diffusion coefficient from a material.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import warnings
from typing import List, Union, Any
import numpy as np
from scipy.stats import normaltest, linregress
from scipy.linalg import pinvh
from scipy.optimize import minimize, curve_fit
import scipy.constants as const
import tqdm
from uravu.distribution import Distribution
from sklearn.utils import resample
from emcee import EnsembleSampler
from kinisi.matrix import find_nearest_positive_definite

DIMENSIONALITY = {
    'x': np.s_[0],
    'y': np.s_[1],
    'z': np.s_[2],
    'xy': np.s_[:2],
    'xz': np.s_[::2],
    'yz': np.s_[1:],
    'xyz': np.s_[:],
    b'x': np.s_[0],
    b'y': np.s_[1],
    b'z': np.s_[2],
    b'xy': np.s_[:2],
    b'xz': np.s_[::2],
    b'yz': np.s_[1:],
    b'xyz': np.s_[:]
}


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

    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 n_o: np.ndarray,
                 sub_sample_dt: int = 1,
                 dimension: str = 'xyz'):
        self._displacements = disp_3d[::sub_sample_dt]
        self._delta_t = np.array(delta_t[::sub_sample_dt])
        self._max_obs = self._displacements[0].shape[1]
        self._distributions = []
        self._dt = np.array([])
        self._n = np.array([])
        self._s = np.array([])
        self._v = np.array([])
        self._n_bootstrap = np.array([])
        self._s_bootstrap = np.array([])
        self._v_bootstrap = np.array([])
        self._n_o = n_o
        self._ngp = np.array([])
        self._sub_sample_dt = sub_sample_dt
        self._dimension = dimension
        self._euclidian_displacements = []
        self._diffusion_coefficient = None
        self._jump_diffusion_coefficient = None
        self._sigma = None
        self._intercept = None
        self.gradient = None
        self.flatchain = None
        self._covariance_matrix = None
        self._slice = DIMENSIONALITY[dimension.lower()]
        self.dims = len(dimension.lower())

    def to_dict(self) -> dict:
        """
        :return: Dictionary description of the :py:class:`Bootstrap`.
        """
        my_dict = {
            'displacements': self._displacements,
            'delta_t': self._delta_t,
            'n_o': self._n_o,
            'max_obs': self._max_obs,
            'dt': self._dt,
            'n': self._n,
            's': self._s,
            'v': self._v,
            'n_bootstrap': self._n_bootstrap,
            's_bootstrap': self._s_bootstrap,
            'v_bootstrap': self._v_bootstrap,
            'sub_sample_dt': self._sub_sample_dt,
            'dimension': self._dimension,
            'ngp': self._ngp,
            'covariance_matrix': self._covariance_matrix,
            'distributions': None,
            'diffusion_coefficient': None,
            'jump_diffusion_coefficient': None,
            'sigma': None,
            'intercept': None,
            'gradient': None
        }
        if len(self._distributions) != 0:
            my_dict['distributions'] = [d.to_dict() for d in self._distributions]
        my_dict['euclidian_displacements'] = [d.to_dict() for d in self._euclidian_displacements]
        if self._diffusion_coefficient is not None:
            my_dict['diffusion_coefficient'] = self._diffusion_coefficient.to_dict()
        if self._jump_diffusion_coefficient is not None:
            my_dict['jump_diffusion_coefficient'] = self._jump_diffusion_coefficient.to_dict()
        if self._sigma is not None:
            my_dict['sigma'] = self._sigma.to_dict()
        if self._intercept is not None:
            my_dict['intercept'] = self._intercept.to_dict()
        if self.gradient is not None:
            my_dict['gradient'] = self.gradient.to_dict()
        my_dict['flatchain'] = self.flatchain
        return my_dict

    @classmethod
    def from_dict(cls, my_dict: dict) -> 'Bootstrap':
        """
        Generate a :py:class:`Bootstrap` object from a dictionary.

        :param my_dict: The input dictionary.

        :return: New :py:class`Bootstrap` object.
        """
        boot = cls(my_dict['delta_t'],
                   my_dict['displacements'],
                   my_dict['n_o'],
                   sub_sample_dt=my_dict['sub_sample_dt'],
                   dimension=my_dict['dimension'])
        boot._max_obs = my_dict['max_obs']
        boot._euclidian_displacements = [Distribution.from_dict(d) for d in my_dict['euclidian_displacements']]
        boot._dt = my_dict['dt']
        boot._n = my_dict['n']
        boot._s = my_dict['s']
        boot._v = my_dict['v']
        boot._n_bootstrap = my_dict['n_bootstrap']
        boot._s_bootstrap = my_dict['s_bootstrap']
        boot._v_bootstrap = my_dict['v_bootstrap']
        boot._n_o = my_dict['n_o']
        boot._ngp = my_dict['ngp']
        if my_dict['distributions'] is not None:
            boot._distributions = [Distribution.from_dict(d) for d in my_dict['distributions']]
        if my_dict['diffusion_coefficient'] is not None:
            boot._diffusion_coefficient = Distribution.from_dict(my_dict['diffusion_coefficient'])
        if my_dict['jump_diffusion_coefficient'] is not None:
            boot._jump_diffusion_coefficient = Distribution.from_dict(my_dict['jump_diffusion_coefficient'])
        if my_dict['sigma'] is not None:
            boot._sigma = Distribution.from_dict(my_dict['sigma'])
        if my_dict['intercept'] is not None:
            boot._intercept = Distribution.from_dict(my_dict['intercept'])
        if my_dict['gradient'] is not None:
            boot.gradient = Distribution.from_dict(my_dict['gradient'])
        boot.flatchain = my_dict['flatchain']
        boot._covariance_matrix = my_dict['covariance_matrix']
        return boot

    @property
    def dt(self) -> np.ndarray:
        """
        :return: Timestep values that were resampled.
        """
        return self._dt

    @property
    def n(self) -> np.ndarray:
        """
        :return: The mean MSD/MSTD/MSCD, as determined from the bootstrap resampling process, in units Å:sup:`2`.
        """
        return self._n

    @property
    def s(self) -> np.ndarray:
        """
        :return: The MSD/MSTD/MSCD standard deviation, as determined from the bootstrap resampling process, in
            units Å:sup:`2`.
        """
        return self._s

    @property
    def v(self) -> np.ndarray:
        """
        :return: The MSD/MSTD/MSCD variance as determined from the bootstrap resampling process, in units Å:sup:`4`.
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
            return tqdm.tqdm(loop, desc='Finding Means and Variances')
        return loop

    @staticmethod
    def sample_until_normal(array: np.ndarray,
                            n_samples: float,
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
        # values = _bayesian_bootstrap(array, n_samples, n_resamples, random_state)
        values = _bootstrap(array, n_samples, n_resamples, random_state)
        p_value = normaltest(values)[1]
        while p_value < alpha and len(values) < max_resamples:
            # values += _bayesian_bootstrap(array, n_samples, 100, random_state)
            values += _bootstrap(array, n_samples, 100, random_state)
            p_value = normaltest(values)[1]
        if len(values) >= max_resamples:
            warnings.warn("The maximum number of resamples has been reached, and the distribution is not yet normal.")
        return Distribution(values)

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
                      thin: int = 10,
                      progress: bool = True,
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
        :param thin: Use only every :py:attr:`thin` samples for the MCMC sampler. Optional, default is :py:attr:`10`.
        :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`.
        :param random_state: A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
            default is :py:attr:`None`.
        """
        if random_state is not None:
            np.random.seed(random_state.get_state()[1][1])

        max_ngp = np.argwhere(self._dt > dt_skip)[0][0]
        if use_ngp:
            max_ngp = np.argmax(self._ngp)

        self._covariance_matrix = self.generate_covariance_matrix(max_ngp)

        _, logdet = np.linalg.slogdet(self._covariance_matrix)
        logdet += np.log(2 * np.pi) * self._n[max_ngp:].size
        inv = pinvh(self._covariance_matrix)

        def log_likelihood(theta: np.ndarray) -> float:
            """
            Get the log likelihood for multivariate normal distribution.
            :param theta: Value of the gradient and intercept of the straight line.
            :return: Log-likelihood value.
            """
            if theta[0] < 0:
                return -np.inf
            model = _straight_line(self._dt[max_ngp:], *theta)
            diff = (model - self._n[max_ngp:])
            logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
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
        self.flatchain = sampler.get_chain(flat=True, thin=thin, discard=n_burn)
        self.gradient = Distribution(self.flatchain[:, 0])
        self._intercept = None
        if fit_intercept:
            self._intercept = Distribution(self.flatchain[:, 1])

    def generate_covariance_matrix(self, max_ngp: int):
        """
        Generate the covariance matrix, including the modelling and finding the closest matrix
        that is positive definite.

        :param max_ngp: The index of the maximum of the non-Gaussian parameter or the point
            where the analysis should begin.
        :return: Modelled covariance matrix that is positive definite.
        """

        def _model_variance(dt: np.ndarray, a: float) -> np.ndarray:
            """
            Determine the model variance, based on a quadratic relationship with the number of
            independent samples as a divisor.

            :param dt: Timestep value
            :param a: Quadratic coefficient
            :return: Model variances
            """
            return a / self._n_o[max_ngp:] * dt**2

        self._popt, _ = curve_fit(_model_variance, self.dt[max_ngp:], self._v[max_ngp:])
        self._model_v = _model_variance(self.dt[max_ngp:], *self._popt)
        self._covariance_matrix = _populate_covariance_matrix(self._model_v, self._n_o[max_ngp:])
        self._npd_covariance_matrix = self._covariance_matrix
        return find_nearest_positive_definite(self._covariance_matrix)

    def diffusion(self, **kwargs):
        """
        Use the bootstrap-GLS method to determine the diffusivity for the system. Keyword arguments will be
        passed of the :py:func:`bootstrap_GLS` method.
        """
        self.bootstrap_GLS(**kwargs)
        self._diffusion_coefficient = Distribution(self.gradient.samples / (2e4 * self.dims))

    @property
    def D(self) -> Union[Distribution, None]:
        """
        An alias for the diffusion coefficient Distribution.

        :return: Diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`.
        """
        return self._diffusion_coefficient

    def jump_diffusion(self, **kwargs):
        """
        Use the bootstrap-GLS method to determine the jump diffusivity for the system. Keyword arguments
        will be passed of the :py:func:`bootstrap_GLS` method.
        """
        self.bootstrap_GLS(**kwargs)
        self._jump_diffusion_coefficient = Distribution(self.gradient.samples /
                                                        (2e4 * self.dims * self._displacements[0].shape[0]))

    @property
    def D_J(self) -> Union[Distribution, None]:
        """
        Alias for the jump diffusion coefficient Distribution.

        :return: Jump diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`.
        """
        return self._jump_diffusion_coefficient

    def conductivity(self, temperature: float, volume: float, **kwargs):
        """
        Use the bootstrap-GLS method to determine the ionic conductivity for the system, in units of mScm:sup:`-1`.
        Keyword arguments will be passed of the :py:func:`bootstrap_GLS` method.

        :param temperature: System temperature, in Kelvin.
        :param volume: System volume, in Å^{3}.
        """
        self.bootstrap_GLS(**kwargs)
        volume = volume * 1e-24  # cm^3
        D = self.gradient.samples / (2e4 * self.dims)  # cm^2s^-1
        conversion = 1000 / (volume * const.N_A) * (const.N_A * const.e)**2 / (const.R * temperature)
        self._sigma = Distribution(D * conversion)

    @property
    def sigma(self) -> Union[Distribution, None]:
        """
        :return: The estimated conductivity, based on the generalised least squares approach, with
            units mScm:sup:`-1`.
        """
        return self._sigma


class MSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the mean
    squared displacements.

    :param delta_t: An array of the timestep values, units of ps
    :param disp_3d: A list of arrays, where each array has the axes
        :code:`[atom, displacement observation, dimension]`. There is one array in the list for each
        delta_t value. Note: it is necessary to use a list of arrays as the number of observations is
        not necessary the same at each data point.
    :param n_o: Number of statistically independent observations of the MSD at each timestep.
    :param sub_sample_dt: The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
    :param bootstrap: Should bootstrap resampling be used to estimate the observed MSD distribution.
        Optional, default is :py:attr:`False`.
    :param n_resamples: The initial number of resamples to be performed. Default is :py:attr:`1000`
    :param max_resamples: The max number of resamples to be performed by the distribution is assumed to be normal.
        This is present to allow user control over the time taken for the resampling to occur. Default
        is :py:attr:`100000`
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param alpha: Value that p-value for the normal test must be greater than to accept. Default is :py:attr:`1e-3`
    :param random_state : A :py:attr:`RandomState` object to be used to ensure reproducibility. Default
        is :py:attr:`None`
    :param progress: Show tqdm progress for sampling. Default is :py:attr:`True`
    """

    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 n_o: np.ndarray,
                 sub_sample_dt: int = 1,
                 bootstrap: bool = False,
                 n_resamples: int = 1000,
                 max_resamples: int = 10000,
                 dimension: str = 'xyz',
                 alpha: float = 1e-3,
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, n_o, sub_sample_dt, dimension)
        self._iterator = self.iterator(progress, range(len(self._displacements)))
        for i in self._iterator:
            disp_slice = self._displacements[i][:, :, self._slice].reshape(self._displacements[i].shape[0],
                                                                           self._displacements[i].shape[1], self.dims)
            d_squared = np.sum(disp_slice**2, axis=-1)
            if d_squared.size <= 1:
                continue
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            if bootstrap:
                distro = self.sample_until_normal(d_squared, n_o[i], n_resamples, max_resamples, alpha, random_state)
                self._distributions.append(distro)
                self._n_bootstrap = np.append(self._n_bootstrap, np.mean(distro.samples))
                self._v_bootstrap = np.append(self._v_bootstrap, np.var(distro.samples, ddof=1))
                self._s_bootstrap = np.append(self._s_bootstrap, np.std(distro.samples, ddof=1))
            self._n = np.append(self._n, d_squared.mean())
            self._v = np.append(self._v, np.var(d_squared, ddof=1) / n_o[i])
            self._s = np.append(self._s, np.sqrt(self._v[i]))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared))
            self._dt = np.append(self._dt, self._delta_t[i])
        self._n_o = self._n_o[:self._n.size]


class MSTDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the total
    mean squared displacements.

    :param delta_t: An array of the timestep values.
    :param disp_3d: A list of arrays, where each array has the axes
        :code:`[atom, displacement observation, dimension]`. There is one array in the list for each
        delta_t value. Note: it is necessary to use a list of arrays as the number of observations is
        not necessary the same at each data point.
    :param n_o: Number of statistically independent observations of the MSD at each timestep.
    :param sub_sample_dt: The frequency in observations to be sampled. Optional, default
        is :py:attr:`1` (every observation).
    :param bootstrap: Should bootstrap resampling be used to estimate the observed MSD distribution.
        Optional, default is :py:attr:`False`.
    :param n_resamples: The initial number of resamples to be performed. Optional, default
        is :py:attr:`1000`
    :param max_resamples: The max number of resamples to be performed by the distribution is assumed to be
        normal. This is present to allow user control over the time taken for the resampling to occur.
        Optional, default is :py:attr:`100000`
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param alpha: Value that p-value for the normal test must be greater than to accept. Optional, default
        is :py:attr:`1e-3`
    :param random_state : A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
        default is :py:attr:`None`
    :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`
    """

    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 n_o: np.ndarray,
                 sub_sample_dt: int = 1,
                 bootstrap: bool = False,
                 n_resamples: int = 1000,
                 max_resamples: int = 10000,
                 dimension: str = 'xyz',
                 alpha: float = 1e-3,
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, n_o, sub_sample_dt, dimension)
        self._iterator = self.iterator(progress, range(int(len(self._displacements) / 2)))
        for i in self._iterator:
            disp_slice = self._displacements[i][:, :, self._slice].reshape(self._displacements[i].shape[0],
                                                                           self._displacements[i].shape[1], self.dims)
            d_squared = np.sum(disp_slice**2, axis=-1)
            coll_motion = np.sum(np.sum(disp_slice, axis=0)**2, axis=-1)
            if coll_motion.size <= 1:
                continue
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            if bootstrap:
                distro = self.sample_until_normal(coll_motion, n_o[i] / d_squared.shape[0], n_resamples, max_resamples, alpha, random_state)
                self._distributions.append(distro)
                self._n_bootstrap = np.append(self._n_bootstrap, np.mean(distro.samples))
                self._v_bootstrap = np.append(self._v_bootstrap, np.var(distro.samples, ddof=1))
                self._s_bootstrap = np.append(self._s_bootstrap, np.std(distro.samples, ddof=1))
            self._n = np.append(self._n, coll_motion.mean())
            self._v = np.append(self._v, np.var(coll_motion, ddof=1) / (n_o[i] / d_squared.shape[0]))
            self._s = np.append(self._s, np.sqrt(self._v[i]))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared.flatten()))
            self._dt = np.append(self._dt, self._delta_t[i])
        self._n_o = self._n_o[:self._n.size]


class MSCDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the mean
    squared charge displacements.

    :param delta_t: An array of the timestep values.
    :param disp_3d: A list of arrays, where each array has the axes
        :code:`[atom, displacement observation, dimension]`. There is one array in the list for each
        delta_t value. Note: it is necessary to use a list of arrays as the number of observations is
        not necessary the same at each data point.
    :param ionic_charge: The charge on the mobile ions, either an array with a value for each ion or a scalar
        if all values are the same.
    :param n_o: Number of statistically independent observations of the MSD at each timestep.
    :param bootstrap: Should bootstrap resampling be used to estimate the observed MSD distribution.
        Optional, default is :py:attr:`False`.
    :param sub_sample_dt: The frequency in observations to be sampled. Optional, default is :py:attr:`1`
        (every observation).
    :param n_resamples: The initial number of resamples to be performed. Optional, default is :py:attr:`1000`.
    :param max_resamples: The max number of resamples to be performed by the distribution is assumed to be normal.
        This is present to allow user control over the time taken for the resampling to occur. Optional, default
        is :py:attr:`100000`.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
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
                 n_o: np.ndarray,
                 sub_sample_dt: int = 1,
                 bootstrap: bool = False,
                 n_resamples: int = 1000,
                 max_resamples: int = 10000,
                 dimension: str = 'xyz',
                 alpha: float = 1e-3,
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, n_o, sub_sample_dt, dimension)
        self._iterator = self.iterator(progress, range(int(len(self._displacements) / 2)))
        try:
            _ = len(ionic_charge)
        except TypeError:
            ionic_charge = np.ones(self._displacements[0].shape[0]) * ionic_charge
        for i in self._iterator:
            disp_slice = self._displacements[i][:, :, self._slice].reshape(self._displacements[i].shape[0],
                                                                           self._displacements[i].shape[1], self.dims)
            d_squared = np.sum(disp_slice**2, axis=-1)
            sq_chg_motion = np.sum(np.sum((ionic_charge * self._displacements[i].T).T, axis=0)**2, axis=-1)
            if sq_chg_motion.size <= 1:
                continue
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            if bootstrap:
                distro = self.sample_until_normal(sq_chg_motion, n_o[i] / d_squared.shape[0], n_resamples, max_resamples, alpha, random_state)
                self._distributions.append(distro)
                self._n_bootstrap = np.append(self._n_bootstrap, np.mean(distro.samples))
                self._v_bootstrap = np.append(self._v_bootstrap, np.var(distro.samples, ddof=1))
                self._s_bootstrap = np.append(self._s_bootstrap, np.std(distro.samples, ddof=1))
            self._n = np.append(self._n, sq_chg_motion.mean())
            self._v = np.append(self._v, np.var(sq_chg_motion, ddof=1) / (n_o[i] / d_squared.shape[0]))
            self._s = np.append(self._s, np.sqrt(self._v[i]))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared.flatten()))
            self._dt = np.append(self._dt, self._delta_t[i])
        self._n_o = self._n_o[:self._n.size]


def _bootstrap(array: np.ndarray,
               n_samples: int,
               n_resamples: float,
               random_state: np.random.mtrand.RandomState = None) -> List[float]:
    """
    Perform a set of resamples.

    :param array: The array to sample from.
    :param n_samples: Number of samples.
    :param n_resamples: Number of resamples to perform.
    :param random_state: A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
        default is :py:attr:`None`

    :return: Simulated means from resampling the array.
    """
    return [
        np.mean(resample(array.flatten(), n_samples=int(n_samples), random_state=random_state).flatten())
        for j in range(n_resamples)
    ]


def _bayesian_bootstrap(array: np.ndarray,
                        n_samples: float,
                        n_resamples: int,
                        random_state: np.random.mtrand.RandomState = None) -> List[float]:
    """
    Performs a Bayesian bootstrap simulation of the posterior distribution of the mean of observed values,
    using a sparse Dirichlet prior for sample weights.
    
    The sparsity of the Dirichlet prior for the sample weights is controlled by a concentration parameter
    alpha, where alpha = k(N-1)/(k-1). k is the dimensionality of the array of observed values, and 
    N can be considered an effective number of samples for each set of sample weights.
    alpha has been chosen to vary linearly with N, and gives a flat Dirichlet prior when N=k,
    and a uniform categorical prior when N=1.
    
    :param array: The array to sample from.
    :param n_samples: The effective number of samples for each set of simulated weights.
    :param n_resamples: Number of resamples to perform.
    :param random_state: A :py:attr:`RandomState` object. Optional, default is :py:attr:`None`
    
    :return: Samples from the simulated posterior distribution of the mean of the array.
    """
    if random_state == None:
        random_state = np.random.mtrand.RandomState()
    values = array.flatten()
    k = len(values)
    alphak = (n_samples - 1) / (k - 1)
    if alphak > 0:
        weights = random_state.dirichlet(alpha=np.ones(k) * alphak, size=n_resamples)
    else:
        # Sample from a uniform categorical distribution, equivalent to Dirichlet([0,0,0,…])
        weights = random_state.multinomial(n=1, pvals=np.ones(k) / k, size=n_resamples)
    return list(np.sum(weights * values, axis=1))


def _populate_covariance_matrix(variances: np.ndarray, n_samples: np.ndarray) -> np.ndarray:
    """
    Populate the covariance matrix for the generalised least squares methodology.

    :param variances: The variances for each timestep
    :param n_samples: Number of independent trajectories for each timestep

    :return: An estimated covariance matrix for the system
    """
    covariance_matrix = np.zeros((variances.size, variances.size))
    for i in range(0, variances.size):
        for j in range(i, variances.size):
            ratio = n_samples[i] / n_samples[j]
            value = ratio * variances[i]
            covariance_matrix[i, j] = value
            covariance_matrix[j, i] = np.copy(covariance_matrix[i, j])
    return covariance_matrix


def _straight_line(abscissa: np.ndarray, gradient: float, intercept: float = 0.0) -> np.ndarray:
    """
    A one dimensional straight line function.

    :param abscissa: The abscissa data.
    :param gradient: The slope of the line.
    :param intercept: The y-intercept of the line. Optional, default is :py:attr:`0.0`.

    :return: The resulting ordinate.
    """
    return gradient * abscissa + intercept
