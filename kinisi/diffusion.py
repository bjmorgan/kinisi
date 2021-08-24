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
import scipy.constants as const
from tqdm import tqdm
from uravu.distribution import Distribution
from sklearn.utils import resample
from emcee import EnsembleSampler
from kinisi.matrix import find_nearest_positive_definite


class Bootstrap:
    """
    The top-level class for bootstrapping.

    Attributes:
        displacements (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        delta_t (:py:attr:`array_like`): An array of the timestep values
        max_obs (:py:attr:`int`): The maximum number of observations for the trajectory
        dt (:py:attr:`array_like`): Timestep values that were resampled
        distributions (:py:attr:`list` of :py:class:`uravu.distribution.Distribution`): Resampled mean squared distributions
        iterator (:py:attr:`tqdm` or :py:attr:`range`): The iteration object
        n (:py:attr:`array_like`): The mean MSD/TMSD/MSCD, as determined from the bootstrap resampling process, in units Å:sup:`2`
        s (:py:attr:`array_like`): The MSD/TMSD/MSCD standard deviation, as determined from the bootstrap resampling process, in units Å:sup:`2`
        v (:py:attr:`array_like`): The MSD/TMSD/MSCD variance as determined from the bootstrap resampling process, in units Å:sup:`4`
        n_i (:py:attr:`array_like`): The number of independent trajectories as a function of :py:attr:`dt`
        ngp (:py:attr:`array_like`): Non-Gaussian parameter as a function of dt
        euclidian_displacements (:py:attr:`list` of :py:class:`uravu.distribution.Distribution`): Displacements between particles at each dt
        diffusion_coefficient (:py:class:`uravu.distribution.Distribution`): The estimated diffusion coefficient, based on the generalised least squares approach
        jump_diffusion_coefficient (:py:class:`uravu.distribution.Distribution`): The estimated jump diffusion coefficient, based on the generalised least squares approach
        sigma (:py:class:`uravu.distribution.Distribution`): The estimated conductivity, based on the generalised least squares approach
        intercept (:py:class:`uravu.distribution.Distribution` or :py:attr:`None`): The estimated intercept. Note that if :py:attr:`fit_intercept` is :py:attr:`False` is the relavent method call, then this is :py:attr:`None`
        covariance_matrix (:py:attr:`array_like`): The covariance matrix for the trajectories

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self, delta_t, disp_3d, sub_sample_dt=1, progress=True):
        self.displacements = disp_3d[::sub_sample_dt]
        self.delta_t = np.array(delta_t[::sub_sample_dt])
        self.max_obs = self.displacements[0].shape[1]
        self.distributions = []
        self.dt = np.array([])
        self.iterator = self.iterator(progress, range(len(self.displacements)))
        self.n = np.array([])
        self.s = np.array([])
        self.v = np.array([])
        self.n_i = np.array([], dtype=int)
        self.ngp = np.array([])
        self.euclidian_displacements = []
        self.diffusion_coefficient = None
        self.jump_diffusion_coeffiecient = None
        self.sigma = None
        self.intercept = None

    @staticmethod
    def iterator(progress, loop):
        """
        Get the iteration object, using :py:mod:`tqdm` as appropriate.

        Args:
            progress (:py:attr:`bool`): Should :py:mod:`tqdm` be used to give a progress bar
            loop (:py:attr:`list` or :py:attr:`range`): The object that should be looped over

        Returns:
            :py:attr:`tqdm` or :py:attr:`range`: Iterator object
        """
        if progress:
            return tqdm(loop, desc='Bootstrapping Displacements')
        return loop

    @staticmethod
    def sample_until_normal(array, n_samples, n_resamples, max_resamples, alpha=1e-3, random_state=None):
        """
        Resample from the distribution until a normal distribution is obtained or a maximum is reached.

        Args:
            array (:py:attr:`array_like`): The array to sample from
            n_samples (:py:attr:`int`): Number of samples
            r_resamples (:py:attr:`int`): Number of resamples to perform initially
            max_resamples (:py:attr:`int`): The maximum number of resamples to perform
            alpha (:py:attr:`float`, optional): Level that p-value should be below in :py:func:`scipy.stats.normaltest` for the distribution to be normal. Default is :py:attr:`1e-3`
            random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:attr:`RandomState` object to be used to ensure reproducibility. Default is :py:attr:`None`

        Returns:
            :py:class:`uravu.distribution.Distribution`: The resampled distribution
        """
        values = _bootstrap(array.flatten(), n_samples, n_resamples, random_state)
        p_value = normaltest(values)[1]
        while (p_value < alpha and len(values) < max_resamples):
            values += _bootstrap(array.flatten(), n_samples, 100, random_state)
        if len(values) >= max_resamples:
            warnings.warn("The maximum number of resamples has been reached, and the distribution is not yet normal.")
        return Distribution(values)

    @staticmethod
    def n_samples(disp_shape, max_obs):
        """
        Calculate the maximum number of independent observations

        Args:
            disp_shape (:py:attr:`tuple`): The shape of the displacements array
            max_obs (:py:attr:`int`): The maximum number of observations for the trajectory

        Returns:
            :py:attr:`int`: Maximum number of independent observations
        """
        n_obs = disp_shape[1]
        n_atoms = disp_shape[0]
        dt_int = max_obs - n_obs + 1
        return int(max_obs / dt_int * n_atoms)

    @staticmethod
    def ngp_calculation(d_squared):
        """
        Determine the non-Gaussian parameter, from S. Song et al, "Transport dynamics of complex fluids" (2019): 10.1073/pnas.1900239116

        Args:
            d_squared (:py:attr:`array_like`): Squared displacement values

        Returns:
            :py:attr:`float`: Value of non-Gaussian parameter
        """
        top = np.mean(d_squared * d_squared) * 3
        bottom = np.square(np.mean(d_squared)) * 5
        return top / bottom - 1

    def bootstrap_GLS(self, use_ngp=False, dt_skip=0, fit_intercept=True, n_walkers=32, n_samples=1000, random_state=None, progress=True):
        """
        Use the covariance matrix estimated from the resampled values to estimate the gradient and intercept using a generalised least squares approach.

        Args:
            use_ngp (:py:attr:`bool`, optional): Should the ngp max be used as the starting point for the diffusion fitting. Default is :py:attr:`False`
            dt_skip (:py:attr:`float`, optional): Values of :py:attr:`dt` that should be skipped, i.e. where the atoms are not diffusing. Note that if :py:attr:`use_ngp` is :py:attr:`True` this will be ignored. Defaults to :py:attr:`0`
            fit_intercept (:py:attr:`bool`, optional): Should the intercept of the diffusion relationship be fit. Default is :py:attr:`True`
            n_walkers (:py:attr:`int`, optional): Number of MCMC walkers to use. Default is :py:attr:`32`
            n_samples (:py:attr:`int`, optional): Number of MCMC samples to perform. Default is :py:attr:`1000`
            random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:attr:`RandomState` object to be used to ensure reproducibility. Default is :py:attr:`None`
            progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
        """
        max_ngp = np.argwhere(self.dt > dt_skip)[0][0]
        if use_ngp:
            max_ngp = np.argmax(self.ngp)
        self.covariance_matrix = self.populate_covariance_matrix(self.v, self.n_i)[max_ngp:, max_ngp:]
        self.covariance_matrix = find_nearest_positive_definite(self.covariance_matrix)

        mv = multivariate_normal(self.n[max_ngp:], self.covariance_matrix, allow_singular=True, seed=random_state)

        def log_likelihood(theta):
            """
            Get the log likelihood for multivariate normal distribution.

            Args:
                theta (:py:attr:`array_like`): Value of the gradient and intercept of the straight line

            Returns:
                (:py:attr:`float`): Log-likelihood value
            """
            model = _straight_line(self.dt[max_ngp:], *theta)
            logl = mv.logpdf(model)
            return logl
        ols = linregress(self.dt[max_ngp:], self.n[max_ngp:])

        def nll(*args):
            """
            General purpose negative log-likelihood.

            Returns:
                (:py:attr:`float`): Negative log-likelihood
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
        self.gradient = Distribution(flatchain[:, 0])
        self.intercept = None
        if fit_intercept:
            self.intercept = Distribution(flatchain[:, 1])

    @staticmethod
    def populate_covariance_matrix(variances, n_samples):
        """
        Populate the covariance matrix for the generalised least squares methodology.

        Args:
            variances (:py:attr:`array_like`): The variances for each timestep
            n_samples (:py:attr:`array_like`): Number of independent trajectories for each timestep

        Returns:
            :py:attr:`array_like`: An estimated covariance matrix for the system
        """
        covariance_matrix = np.zeros((variances.size, variances.size))
        for i in range(0, variances.size):
            for j in range(i, variances.size):
                ratio = n_samples[i] / n_samples[j]
                covariance_matrix[i, j] = variances[i] * ratio
                covariance_matrix[j, i] = np.copy(covariance_matrix[i, j])
        return covariance_matrix


class MSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the mean squared displacements.

    Attributes:
        n (:py:attr:`array_like`): The mean MSD, as determined from the bootstrap resampling process
        s (:py:attr:`array_like`): The MSD standard deviation, as determined from the bootstrap resampling process
        v (:py:attr:`array_like`): The MSD variance as determined from the bootstrap resampling process
        diffusion_coefficient (:py:class:`uravu.distribution.Distribution`): The estimated diffusion coefficient, based on the generalised least squares approach, with units of cm:sup:`2`s:sup:`-1`
        intercept (:py:class:`uravu.distribution.Distribution` or :py:attr:`None`): The estimated intercept, with units of Å:sup:`2`ps:sup:`-1`. Note that if :py:attr:`fit_intercept` is :py:attr:`False` is the relavent method call, then this is :py:attr:`None`

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values, units of ps
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`
        alpha (:py:attr:`float`, optional): Value that p-value for the normal test must be greater than to accept. Default is :py:attr:`1e-3`
        random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:attr:`RandomState` object to be used to ensure reproducibility. Default is :py:attr:`None`
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self, delta_t, disp_3d, sub_sample_dt=1, n_resamples=1000, max_resamples=10000, alpha=1e-3, random_state=None, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, progress)
        for i in self.iterator:
            d_squared = np.sum(self.displacements[i] ** 2, axis=2)
            n_samples_current = self.n_samples(self.displacements[i].shape, self.max_obs)
            if n_samples_current <= 1:
                continue
            self.n_i = np.append(self.n_i, n_samples_current)
            self.euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            distro = self.sample_until_normal(d_squared, self.n_i[i], n_resamples, max_resamples, alpha, random_state)
            self.distributions.append(distro)
            self.n = np.append(self.n, distro.n)
            self.s = np.append(self.s, np.std(distro.samples, ddof=1))
            self.v = np.append(self.v, np.var(distro.samples, ddof=1))
            self.ngp = np.append(self.ngp, self.ngp_calculation(d_squared))
            self.dt = np.append(self.dt, self.delta_t[i])

    def diffusion(self, **kwargs):
        """
        Use the bootstrap-GLS method to determine the diffusivity for the system.
        """
        self.bootstrap_GLS(**kwargs)
        self.diffusion_coefficient = Distribution(self.gradient.samples / (2e4 * self.displacements[0].shape[-1]))

    @property
    def D(self):
        """
        An alias for the diffusion coefficient Distribution.

        Returns:
            (:py:class:`uravu.distribution.Distribution`): Diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`
        """
        return self.diffusion_coefficient


class TMSDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the total mean squared displacements.

    Attributes:
        n (:py:attr:`array_like`): The mean TMSD, as determined from the bootstrap resampling process
        s (:py:attr:`array_like`): The TMSD standard deviation, as determined from the bootstrap resampling process
        v (:py:attr:`array_like`): The TMSD variance as determined from the bootstrap resampling process
        jump_diffusion_coefficient (:py:class:`uravu.distribution.Distribution`): The estimated jump diffusion coefficient, based on the generalised least squares approach, with units of cm:sup:`2`s:sup:`-1`
        intercept (:py:class:`uravu.distribution.Distribution` or :py:attr:`None`): The estimated intercept, with units of Å:sup:`2`ps:sup:`-1`. Note that if :py:attr:`fit_intercept` is :py:attr:`False` is the relavent method call, then this is :py:attr:`None`

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`
        alpha (:py:attr:`float`, optional): Value that p-value for the normal test must be greater than to accept. Default is :py:attr:`1e-3`
        random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:attr:`RandomState` object to be used to ensure reproducibility. Default is :py:attr:`None`
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self, delta_t, disp_3d, sub_sample_dt=1, n_resamples=1000, max_resamples=10000, alpha=1e-3, random_state=None, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, progress)
        for i in self.iterator:
            d_squared = np.sum(self.displacements[i] ** 2, axis=2)
            coll_motion = np.sum(np.sum(self.displacements[i], axis=0) ** 2, axis=-1)
            n_samples_current = self.n_samples((1, self.displacements[i].shape[1]), self.max_obs)
            if n_samples_current <= 1:
                continue
            self.n_i = np.append(self.n_i, n_samples_current)
            self.euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            distro = self.sample_until_normal(coll_motion, self.n_i[i], n_resamples, max_resamples, alpha, random_state)
            self.distributions.append(distro)
            self.n = np.append(self.n, distro.n)
            self.s = np.append(self.s, np.std(distro.samples, ddof=1))
            self.v = np.append(self.v, np.var(distro.samples, ddof=1))
            self.ngp = np.append(self.ngp, self.ngp_calculation(d_squared.flatten()))
            self.dt = np.append(self.dt, self.delta_t[i])

    def jump_diffusion(self, **kwargs):
        """
        Use the bootstrap-GLS method to determine the jump diffusivity for the system.
        """
        self.bootstrap_GLS(**kwargs)
        self.jump_diffusion_coefficient = Distribution(self.gradient.samples / (2e4 * self.displacements[0].shape[-1] * self.displacements[0].shape[0]))

    @property
    def D_J(self):
        """
        Alias for the jump diffusion coefficient Distribution.

        Returns:
            (:py:class:`uravu.distribution.Distribution`): Jump diffusion coefficient, with units of cm:sup:`2`s:sup:`-1`
        """
        return self.jump_diffusion_coefficient


class MSCDBootstrap(Bootstrap):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean and uncertainty for the mean squared charge displacements.

    Attributes:
        n (:py:attr:`array_like`): The mean MSCD, as determined from the bootstrap resampling process
        s (:py:attr:`array_like`): The MSCD standard deviation, as determined from the bootstrap resampling process
        v (:py:attr:`array_like`): The MSCD variance as determined from the bootstrap resampling process
        sigma (:py:class:`uravu.distribution.Distribution`): The estimated conductivity, based on the generalised least squares approach, with units mScm:sup:`-1`
        intercept (:py:class:`uravu.distribution.Distribution` or :py:attr:`None`): The estimated intercept, with units of Å:sup:`2`ps:sup:`-1`. Note that if :py:attr:`fit_intercept` is :py:attr:`False` is the relavent method call, then this is :py:attr:`None`

    Args:
        delta_t (:py:attr:`array_like`): An array of the timestep values
        disp_3d (:py:attr:`list` of :py:attr:`array_like`): A list of arrays, where each array has the axes [atom, displacement observation, dimension]. There is one array in the list for each delta_t value. Note: it is necessary to use a list of arrays as the number of observations is not necessary the same at each data point
        ionic_charge (:py:attr:`array_like` or :py:attr:`int`): The charge on the mobile ions, either an array with a value for each ion or a scalar if all values are the same
        sub_sample_dt (:py:attr:`int`. optional): The frequency in observations to be sampled. Default is :py:attr:`1` (every observation)
        n_resamples (:py:attr:`int`, optional): The initial number of resamples to be performed. Default is :py:attr:`1000`
        max_resamples (:py:attr:`int`, optional): The max number of resamples to be performed by the distribution is assumed to be normal. This is present to allow user control over the time taken for the resampling to occur. Default is :py:attr:`100000`
        alpha (:py:attr:`float`, optional): Value that p-value for the normal test must be greater than to accept. Default is :py:attr:`1e-3`
        random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:attr:`RandomState` object to be used to ensure reproducibility. Default is :py:attr:`None`
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`
    """
    def __init__(self, delta_t, disp_3d, ionic_charge, sub_sample_dt=1, n_resamples=1000, max_resamples=10000, alpha=1e-3, random_state=None, progress=True):
        super().__init__(delta_t, disp_3d, sub_sample_dt, progress)
        try:
            _ = len(ionic_charge)
        except TypeError:
            ionic_charge = np.ones(self.displacements[0].shape[0]) * ionic_charge
        for i in self.iterator:
            d_squared = np.sum(self.displacements[i] ** 2, axis=2)
            sq_chg_motion = np.sum(np.sum((ionic_charge * self.displacements[i].T).T, axis=0) ** 2, axis=-1)
            n_samples_current = self.n_samples((1, self.displacements[i].shape[1]), self.max_obs)
            if n_samples_current <= 1:
                continue
            self.n_i = np.append(self.n_i, n_samples_current)
            self.euclidian_displacements.append(Distribution(np.sqrt(d_squared.flatten())))
            distro = self.sample_until_normal(sq_chg_motion, self.n_i[i], n_resamples, max_resamples, alpha, random_state)
            self.distributions.append(distro)
            self.n = np.append(self.n, distro.n)
            self.s = np.append(self.s, np.std(distro.samples, ddof=1))
            self.v = np.append(self.v, np.var(distro.samples, ddof=1))
            self.ngp = np.append(self.ngp, self.ngp_calculation(d_squared.flatten()))
            self.dt = np.append(self.dt, self.delta_t[i])

    def conductivity(self, temperature, volume, **kwargs):
        """
        Use the bootstrap-GLS method to determine the ionic conductivity for the system, in units of mScm:sup:`-1`.

        Args:
            temperature (:py:attr:`float`): System temperature, in Kelvin
            volume (:py:attr:`float`): System volume, in Å^{3}
        """
        self.bootstrap_GLS(**kwargs)
        volume = volume * 1e-24 # cm^3
        D = self.gradient.samples / (2e4 * self.displacements[0].shape[-1])  # cm^2s^-1
        conversion = 1000 / (volume * const.N_A) * (const.N_A * const.e) ** 2 / (const.R * temperature)
        self.sigma = Distribution(D * conversion)


def _bootstrap(array, n_samples, n_resamples, random_state=None):
    """
    Perform a set of resamples.

    Args:
        array (:py:attr:`array_like`): The array to sample from
        n_samples (:py:attr:`int`): Number of samples
        n_resamples (:py:attr:`int`): Number of resamples to perform initially
        random_state (:py:class:`numpy.random.mtrand.RandomState`, optional): A :py:attr:`RandomState` object to be used to ensure reproducibility. Default is :py:attr:`None`

    Returns:
        :py:attr:`array_like`: Resampled values from the array
    """
    return [np.mean(resample(array.flatten(), n_samples=n_samples, random_state=random_state)) for j in range(n_resamples)]


def _straight_line(abscissa, gradient, intercept=0):
    """
    A one dimensional straight line function.

    Args:
        abscissa (:py:attr:`array_like`): The abscissa data
        gradient (:py:attr:`float`): The slope of the line
        intercept (:py:attr:`float`): The y-intercept of the line

    Returns:
        :py:attr:`array_like`: The resulting ordinate
    """
    return gradient * abscissa + intercept
