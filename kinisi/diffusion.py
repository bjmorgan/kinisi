"""
The modules is focused on tools for the evaluation of the mean squared displacement and resulting
diffusion coefficient from a material.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import warnings
from typing import List, Union
import numpy as np
from scipy.stats import normaltest, linregress, multivariate_normal
from scipy.linalg import pinvh
from scipy.optimize import minimize, curve_fit
import scipy.constants as const
import tqdm
from uravu.distribution import Distribution
from emcee import EnsembleSampler
from statsmodels.stats.correlation_tools import cov_nearest

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
        self._hr = np.array([])
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
        self._start_dt = None
        self.dims = len(dimension.lower())
        self._model = None

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
            'hr': self._hr,
            'sub_sample_dt': self._sub_sample_dt,
            'dimension': self._dimension,
            'ngp': self._ngp,
            'covariance_matrix': self._covariance_matrix,
            'distributions': None,
            'diffusion_coefficient': None,
            'jump_diffusion_coefficient': None,
            'sigma': None,
            'intercept': None,
            'gradient': None,
            'start_dt': self._start_dt,
            'model': self._model
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
        boot._hr = my_dict['hr']
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
        boot._start_dt = my_dict['start_dt']
        boot._model = my_dict['model']
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
                      start_dt: float,
                      model: bool = True,
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

        :param start_dt: The starting time for the analysis to find the diffusion coefficient.
            This should be the start of the diffusive regime in the simulation.
        :param model: Use the model for the covariance matrix, if False this may lead to numerical instability.
            Optional, default is :py:attr:`True`.
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

        self._start_dt = start_dt
        self._model = model

        diff_regime = np.argwhere(self._dt >= self._start_dt)[0][0]
        self._v *= self._n_o
        self._n_o *= self._hr[diff_regime]
        self._v /= self._n_o
        self._s = np.sqrt(self._v)

        self._covariance_matrix = self.generate_covariance_matrix(diff_regime)

        _, logdet = np.linalg.slogdet(self._covariance_matrix)
        logdet += np.log(2 * np.pi) * self._n[diff_regime:].size
        inv = pinvh(self._covariance_matrix)

        def log_likelihood(theta: np.ndarray) -> float:
            """
            Get the log likelihood for multivariate normal distribution.
            :param theta: Value of the gradient and intercept of the straight line.
            :return: Log-likelihood value.
            """
            if theta[0] < 0:
                return -np.inf
            model = _straight_line(self._dt[diff_regime:], *theta)
            diff = (model - self._n[diff_regime:])
            logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
            return logl

        ols = linregress(self._dt[diff_regime:], self._n[diff_regime:])
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

    def generate_covariance_matrix(self, diff_regime: int):
        """
        Generate the covariance matrix, including the modelling and finding the closest matrix
        that is positive definite.

        :param diff_regime: The index of the point where the analysis should begin.
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
            return a / self._n_o[diff_regime:] * dt**2

        if self._model:
            self._popt, _ = curve_fit(_model_variance, self.dt[diff_regime:], self._v[diff_regime:])
            self._model_v = _model_variance(self.dt[diff_regime:], *self._popt)
        else:
            self._model_v = self._v[diff_regime:]
        self._covariance_matrix = _populate_covariance_matrix(self._model_v, self._n_o[diff_regime:])
        self._npd_covariance_matrix = self._covariance_matrix
        return cov_nearest(self._covariance_matrix)

    def diffusion(self, start_dt: float, **kwargs):
        """
        Use the bootstrap-GLS method to determine the diffusivity for the system. Keyword arguments will be
        passed of the :py:func:`bootstrap_GLS` method.

        :param start_dt: The starting time for the analysis to find the diffusion coefficient.
            This should be the start of the diffusive regime in the simulation.
        """
        self.bootstrap_GLS(start_dt, **kwargs)
        self._diffusion_coefficient = Distribution(self.gradient.samples / (2e4 * self.dims))

    @property
    def D(self) -> Union[Distribution, None]:
        """
        An alias for the diffusion coefficient Distribution.

        :return: Diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`.
        """
        return self._diffusion_coefficient

    def jump_diffusion(self, start_dt: float, **kwargs):
        """
        Use the bootstrap-GLS method to determine the jump diffusivity for the system. Keyword arguments
        will be passed of the :py:func:`bootstrap_GLS` method.

        :param start_dt: The starting time for the analysis to find the diffusion coefficient.
            This should be the start of the diffusive regime in the simulation.
        """
        self.bootstrap_GLS(start_dt, **kwargs)
        self._jump_diffusion_coefficient = Distribution(self.gradient.samples /
                                                        (2e4 * self.dims * self._displacements[0].shape[0]))

    @property
    def D_J(self) -> Union[Distribution, None]:
        """
        Alias for the jump diffusion coefficient Distribution.

        :return: Jump diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`.
        """
        return self._jump_diffusion_coefficient

    def conductivity(self, start_dt: float, temperature: float, volume: float, **kwargs):
        """
        Use the bootstrap-GLS method to determine the ionic conductivity for the system, in units of mScm:sup:`-1`.
        Keyword arguments will be passed of the :py:func:`bootstrap_GLS` method.

        :param start_dt: The starting time for the analysis to find the diffusion coefficient.
            This should be the start of the diffusive regime in the simulation.
        :param temperature: System temperature, in Kelvin.
        :param volume: System volume, in Å^{3}.
        """
        self.bootstrap_GLS(start_dt, **kwargs)
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

    def posterior_predictive(self,
                             n_posterior_samples: int = None,
                             n_predictive_samples: int = 256,
                             progress: bool = True) -> np.ndarray:
        """
        Sample the posterior predictive distribution. The shape of the resulting array will be
        `(n_posterior_samples * n_predictive_samples, start_dt)`.
        
        :param n_posterior_samples: Number of samples from the posterior distribution.
            Optional, default is the number of posterior samples.
        :param n_predictive_samples: Number of random samples per sample from the posterior distribution.
            Optional, default is :py:attr:`256`.
        :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`.

        :return: Samples from the posterior predictive distribution. 
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.gradient.size
        diff_regime = np.argwhere(self._dt >= self._start_dt)[0][0]
        ppd = np.zeros((n_posterior_samples, n_predictive_samples, self._dt[diff_regime:].size))
        samples_to_draw = list(enumerate(np.random.randint(0, self.gradient.size, size=n_posterior_samples)))
        if progress:
            iterator = tqdm.tqdm(samples_to_draw, desc='Calculating Posterior Predictive')
        else:
            iterator = samples_to_draw
        for i, n in iterator:
            mu = self.gradient.samples[n] * self._dt[diff_regime:] + self.intercept.samples[n]
            mv = multivariate_normal(mean=mu, cov=self._covariance_matrix)
            ppd[i] = mv.rvs(n_predictive_samples)
        return ppd.reshape(-1, self._dt[diff_regime:].size)


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
    :param block: Should the blocking method be used to estimate the variance, if :py:attr:`False` an 
        approximation is used to estimate the variance. Optional, default is :py:attr:`False`.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param random_state : A :py:attr:`RandomState` object to be used to ensure reproducibility. Default
        is :py:attr:`None`
    :param progress: Show tqdm progress for sampling. Default is :py:attr:`True`
    """

    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 n_o: np.ndarray,
                 sub_sample_dt: int = 1,
                 block: bool = False,
                 dimension: str = 'xyz',
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, n_o, sub_sample_dt, dimension)
        self._iterator = self.iterator(progress, range(len(self._displacements)))
        if block:
            import pyblock
            print('You are using the blocking method to estimate variances, please cite '
                  'doi:10.1063/1.457480 and the pyblock package.')
        for i in self._iterator:
            disp_slice = self._displacements[i][:, :, self._slice].reshape(self._displacements[i].shape[0],
                                                                           self._displacements[i].shape[1], self.dims)
            d_squared = np.sum(disp_slice**2, axis=-1)
            coll_motion = np.sum(np.sum(disp_slice, axis=0)**2, axis=-1) / disp_slice.shape[0]
            hr = d_squared.mean() / coll_motion.mean()
            self._hr = np.append(self._hr, hr)
            if d_squared.size <= 1:
                continue
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            if block:
                reblock = pyblock.blocking.reblock(d_squared.flatten())
                opt_block = pyblock.blocking.find_optimal_block(d_squared.flatten().size, reblock)
                try:
                    mean = reblock[opt_block[0]].mean
                    var = reblock[opt_block[0]].std_err**2
                except TypeError:
                    mean = reblock[-1].mean
                    var = reblock[-1].std_err**2
                self._n = np.append(self._n, mean)
                self._v = np.append(self._v, var)
                self._n_o[i] = 1
            else:
                self._n = np.append(self._n, d_squared.mean())
                self._v = np.append(self._v, np.var(d_squared, ddof=1))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared))
            self._dt = np.append(self._dt, self._delta_t[i])
        self._n_o = self._n_o[:self._n.size]
        self._v /= self._n_o
        self._s = np.sqrt(self._v)


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
    :param block: Should the blocking method be used to estimate the variance, if :py:attr:`False` an 
        approximation is used to estimate the variance. Optional, default is :py:attr:`False`.
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
    :param random_state : A :py:attr:`RandomState` object to be used to ensure reproducibility. Optional,
        default is :py:attr:`None`
    :param progress: Show tqdm progress for sampling. Optional, default is :py:attr:`True`
    """

    def __init__(self,
                 delta_t: np.ndarray,
                 disp_3d: List[np.ndarray],
                 n_o: np.ndarray,
                 sub_sample_dt: int = 1,
                 block: bool = False,
                 dimension: str = 'xyz',
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, n_o, sub_sample_dt, dimension)
        self._iterator = self.iterator(progress, range(int(len(self._displacements) / 2)))
        if block:
            import pyblock
            print('You are using the blocking method to estimate variances, please cite '
                  'doi:10.1063/1.457480 and the pyblock package.')
        for i in self._iterator:
            disp_slice = self._displacements[i][:, :, self._slice].reshape(self._displacements[i].shape[0],
                                                                           self._displacements[i].shape[1], self.dims)
            d_squared = np.sum(disp_slice**2, axis=-1)
            coll_motion = np.sum(np.sum(disp_slice, axis=0)**2, axis=-1)
            self._hr = np.append(self._hr, 1)
            if coll_motion.size <= 1:
                continue
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            if block:
                reblock = pyblock.blocking.reblock(coll_motion.flatten())
                opt_block = pyblock.blocking.find_optimal_block(coll_motion.flatten().size, reblock)
                try:
                    mean = reblock[opt_block[0]].mean
                    var = reblock[opt_block[0]].std_err**2
                except TypeError:
                    mean = reblock[-1].mean
                    var = reblock[-1].std_err**2
                self._n = np.append(self._n, mean)
                self._v = np.append(self._v, var)
                self._n_o[i] = 1
            else:
                self._n = np.append(self._n, coll_motion.mean())
                self._v = np.append(self._v, np.var(coll_motion, ddof=1) / (d_squared.shape[0]))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared.flatten()))
            self._dt = np.append(self._dt, self._delta_t[i])
        self._n_o = self._n_o[:self._n.size]
        self._v /= self._n_o
        self._s = np.sqrt(self._v)


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
    :param block: Should the blocking method be used to estimate the variance, if :py:attr:`False` an 
        approximation is used to estimate the variance. Optional, default is :py:attr:`False`.
    :param sub_sample_dt: The frequency in observations to be sampled. Optional, default is :py:attr:`1`
        (every observation).
    :param dimension: Dimension/s to find the displacement along, this should be some subset of `'xyz'` indicating
        the axes of interest. Optional, defaults to `'xyz'`.
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
                 block: bool = False,
                 dimension: str = 'xyz',
                 random_state: np.random.mtrand.RandomState = None,
                 progress: bool = True):
        super().__init__(delta_t, disp_3d, n_o, sub_sample_dt, dimension)
        self._iterator = self.iterator(progress, range(int(len(self._displacements) / 2)))
        try:
            _ = len(ionic_charge)
        except TypeError:
            ionic_charge = np.ones(self._displacements[0].shape[0]) * ionic_charge
        if block:
            import pyblock
            print('You are using the blocking method to estimate variances, please cite '
                  'doi:10.1063/1.457480 and the pyblock package.')
        for i in self._iterator:
            disp_slice = self._displacements[i][:, :, self._slice].reshape(self._displacements[i].shape[0],
                                                                           self._displacements[i].shape[1], self.dims)
            d_squared = np.sum(disp_slice**2, axis=-1)
            sq_chg_motion = np.sum(np.sum((ionic_charge * self._displacements[i].T).T, axis=0)**2, axis=-1)
            self._hr = np.append(self._hr, 1)
            if sq_chg_motion.size <= 1:
                continue
            self._euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            if block:
                reblock = pyblock.blocking.reblock(sq_chg_motion.flatten())
                opt_block = pyblock.blocking.find_optimal_block(sq_chg_motion.flatten().size, reblock)
                try:
                    mean = reblock[opt_block[0]].mean
                    var = reblock[opt_block[0]].std_err**2
                except TypeError:
                    mean = reblock[-1].mean
                    var = reblock[-1].std_err**2
                self._n = np.append(self._n, mean)
                self._v = np.append(self._v, var)
                self._n_o[i] = 1
            else:
                self._n = np.append(self._n, sq_chg_motion.mean())
                self._v = np.append(self._v, np.var(sq_chg_motion, ddof=1) / (d_squared.shape[0]))
            self._ngp = np.append(self._ngp, self.ngp_calculation(d_squared.flatten()))
            self._dt = np.append(self._dt, self._delta_t[i])
        self._n_o = self._n_o[:self._n.size]
        self._v /= self._n_o
        self._s = np.sqrt(self._v)


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
