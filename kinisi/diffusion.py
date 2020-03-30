"""
The modules is focused on tools for the evaluation of the mean squared
displacement and resulting diffusion coefficient from a material.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913,R0914

import sys
import warnings
import numpy as np
from scipy.stats import uniform
from sklearn.utils import resample
from tqdm import tqdm
from uravu.distribution import Distribution
from uravu.relationship import Relationship
from uravu import UREG, utils


def msd_bootstrap(delta_t, disp_3d, n_resamples=1000, samples_freq=1,
                  confidence_interval=None, max_resamples=100000,
                  bootstrap_multiplier=1, progress=True):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean
    and uncertainty for the squared displacements. This resampling method is
    applied until the MSD distribution is normal(or the `max_resamples` has
    been reached) and therefore may be described with a median and confidence
    interval.

    Args:
        delta_t (array_like): An array of the timestep values.
        displacements (list of array_like): A list of arrays, where
            each array has the axes [atom, displacement
            observation, dimension]. There is one array in the list for each
            delta_t value. Note: it is necessary to use a list of arrays
            as the number of observations is not necessary the same at
            each data point.
        n_resamples (int, optional): The initial number of resamples to
            be performed. Default is `1000`.
        samples_freq (int. optional): The frequency in observations to be
            sampled. Default is `1` (every observation).
        confidence_interval (array_like): The percentile points of the
            distribution that should be stored. Default is `[2.5, 97.5]` (a
            95 % confidence interval).
        max_resamples (int, optional): The max number of resamples to be
            performed by the distribution is assumed to be normal. This is
            present to allow user control over the time taken for the
            resampling to occur. Default is `100000`.
        bootstrap_multiplier (int, optional): The factor by which the number
            of bootstrap samples should be multiplied. The default is `1`,
            which is the maximum number of truely independent samples in a
            given timestep. This can be increase, however it is importance
            to note that when greater than 1 the sampling is no longer
            independent.
        progress (bool, optional): Show tqdm progress for sampling. Default
            is `True`.

    Returns:
        (tuple of array_like) A tuple of five arrays, the first is the delta_t
            values that were resampled, the second is the resampled mean
            squared displacement data, the third is the standard deviation in
            the mean squared displacement, the fourth is the lower bound on
            the confidence interval, and the fifth is the upper bound on the
            confidence interval.
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
            err_msd, np.std(distro.samples))
        con_int_msd_lower = np.append(con_int_msd_lower, distro.con_int[0])
        con_int_msd_upper = np.append(con_int_msd_upper, distro.con_int[1])
    return (
        output_delta_t, mean_msd, err_msd, con_int_msd_lower,
        con_int_msd_upper)


def mscd_bootstrap(delta_t, displacements, indices=None, n_resamples=1000,
                   samples_freq=1, confidence_interval=None,
                   max_resamples=100000, bootstrap_multiplier=1,
                   progress=True):
    """
    Perform a bootstrap resampling to obtain accurate estimates for the mean
    and uncertainty for the squared charge displacements. This
    resampling method is applied until the MSCD distribution is normal(or the
    `max_resamples` has been reached) and therefore may be described with a
    median and confidence interval.

    Args:
        delta_t (array_like): An array of the timestep values.
        displacements (list of array_like): A list of arrays, where
            each array has the axes [atom, displacement
            observation, dimension]. There is one array in the list for each
            delta_t value. Note: it is necessary to use a list of arrays
            as the number of observations is not necessary the same at
            each data point.
        indices (array_like, optional): The indices of the particles of
            interest for charge displacement. Default to all particles.
        n_resamples (int, optional): The initial number of resamples to
            be performed. Default is `1000`.
        samples_freq (int. optional): The frequency in observations to be
            sampled. Default is `1` (every observation).
        confidence_interval (array_like): The percentile points of the
            distribution that should be stored. Default is `[2.5, 97.5]` (a
            95 % confidence interval).
        max_resamples (int, optional): The max number of resamples to be
            performed by the distribution is assumed to be normal. This is
            present to allow user control over the time taken for the
            resampling to occur. Default is `100000`.
        bootstrap_multiplier (int, optional): The factor by which the number
            of bootstrap samples should be multiplied. The default is `1`,
            which is the maximum number of truely independent samples in a
            given timestep. This can be increase, however it is importance
            to note that when greater than 1 the sampling is no longer
            independent.
        progress (bool, optional): Show tqdm progress for sampling. Default
            is `True`.

    Returns:
        (tuple of array_like) A tuple of five arrays, the first is the delta_t
            values that were resampled, the second is the resampled mean
            squared charge displacement data, the third is the standard
            deviation in the mean squared charge displacement, the fourth is
            the lower bound on the confidence interval, and the fifth is the
            upper bound on the confidence interval.
    """
    if confidence_interval is None:
        confidence_interval = [2.5, 97.5]
    max_obs = displacements[0].shape[1]
    output_delta_t = np.array([])
    mean_mscd = np.array([])
    err_mscd = np.array([])
    con_int_mscd_lower = np.array([])
    con_int_mscd_upper = np.array([])
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
        err_mscd = np.append(err_mscd, np.std(distro.samples))
        con_int_mscd_lower = np.append(
            con_int_mscd_lower, distro.con_int[0] / len(indices))
        con_int_mscd_upper = np.append(
            con_int_mscd_upper, distro.con_int[1] / len(indices))
    return (
        output_delta_t, mean_mscd, err_mscd, con_int_mscd_lower,
        con_int_mscd_upper)


class Diffusion(Relationship):
    """
    Evaluate the diffusion coefficient from a set of (idelly) resampled MSD
    data and delta_t values.

    Attributes:
        diffusion_coefficient (uncertainties.ufloat or
            kinisi.distribution.Distribution): The value and associated
            uncertainty for the diffusion coeffcient for the MSD relationship
            with delta_t. The uncertainty is initially obtained from a
            weighted least squares fit, the accuracy of this can be improved
            by using the `sample()` method. The unit is
            centimeter ** 2 / second.
    """
    def __init__(self, delta_t, msd, msd_error,
                 delta_t_unit=UREG.femtoseconds, msd_unit=UREG.angstrom**2,
                 delta_t_names=r'$\delta t$',
                 msd_names=r'$\langle r ^ 2 \rangle$', unaccounted_uncertainty=False):
        variable_names = ['m', r'$D_0$']
        variable_units = [msd_unit / delta_t_unit, msd_unit]
        if unaccounted_uncertainty:
            variable_names.append(r'$f$')
            variable_units.append([msd_unit])
        super().__init__(
            utils.straight_line, delta_t, msd, msd_error, delta_t_unit, msd_unit, delta_t_names,
            msd_names, variable_names, variable_units, unaccounted_uncertainty)

    @property
    def diffusion_coefficient(self):
        """
        Get the diffusion coefficient. 

        Returns:
            (float or Distribution): The diffusion coefficient in the input units.
        """
        if isinstance(self.variables[0], Distribution):
            return Distribution(self.variables[0].samples / 6, r'$D$', None, self.variable_units[0])
        else:
            return self.variables[0] / 6 * self.variable_units[0]

    def sample(self, **kwargs):
        """
        Use MCMC to sample the posterior distribution of the linear diffusion relationship.
        """
        def all_positive_prior():
            """
            This creates an all positive prior.
            """
            priors = []
            for var in self.variable_medians:
                loc = (var - np.abs(var) * 5)
                scale = (var + np.abs(var) * 5) - loc
                priors.append(uniform(loc=loc, scale=scale))
            if self.unaccounted_uncertainty:
                priors[-1] = uniform(loc=-10, scale=11)
            return priors
        self.mcmc(prior_function=all_positive_prior, **kwargs)
