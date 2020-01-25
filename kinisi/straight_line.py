"""
Functions related to the fitting and sampling of a straight line.
"""
# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913

import numpy as np
import emcee
from scipy.optimize import curve_fit
from uncertainties import ufloat
from kinisi import utils


def straight_line(abscissa, gradient, intercept):
    """
    Calculate ordinate of straight line

    Args:
        abscissa (array_like): abscissa
        gradient (float): gradient
        intercept (float): intercept

    Returns:
        (array_like) ordinate
    """
    return gradient * abscissa + intercept


def equation(abscissa, ordinate, uncertainty):
    """
    Estimate the straight line gradient and intercept

    Returns:
        (tuple) Length of 2, gradient and intercept.
    """
    popt, pcov = curve_fit(
        straight_line,
        abscissa,
        ordinate,
        sigma=uncertainty,
    )
    perr = np.sqrt(np.diag(pcov))
    gradient = ufloat(popt[0], perr[0])
    intercept = ufloat(popt[1], perr[1])
    return gradient, intercept


def prior(initial_guess, size):
    """
    Generate initialisation guess

    Args:
        initial_guess (float): Initial value to generate prior from.
        size (int): Number of walkers from MCMC.

    Returns:
        (array_like) Uniform prior centred on initial_guess.
    """
    minimum = initial_guess - 10. * initial_guess
    maximum = initial_guess + 10. * initial_guess

    return np.random.uniform(minimum, maximum, size)


def comparision(theta, y_data, dy_data, x_data):
    """
    Generate model data and get logl.

    Args:
        theta (tuple): Values for variables.
        y_data (array_like): Experimental ordinate data.
        dy_data (array_like): Experimental ordinate-uncertainty data.
        x_data(array_like): Experimental abscissa data.
    Returns:
        (float) ln-likelihood between model and data.
    """
    gradient, intercept = theta

    model = straight_line(x_data, gradient, intercept)

    return utils.lnl(model, y_data, dy_data)


def run_sampling(init_guesses, y_data, dy_data, x_data, walkers=100,
                 n_samples=500, n_burn=500, progress=True):
    """
    Perform MCMC to get gradient and intercept.

    Args:
        init_guesses (tuple): initial guesses for gradient and intercept
            respectivily.
        y_data (array_like): Experimental ordinate data.
        dy_data (array_like): Experimental ordinate-uncertainty data.
        x_data(array_like): Experimental abscissa data.
        walkers (int, optional): Number of MCMC walkers.
        n_samples (int, optional): Number of sample points.
        n_burn (int, optional): Number of burn in samples.
        progress (bool, optional): Show tqdm progress for sampling.

    Returns:
        (array_like) Samples for the variables from MCMC
    """
    gradient_prior = prior(init_guesses[0].n, walkers)
    intercept_prior = prior(init_guesses[1].n, walkers)
    initial_prior = np.array([gradient_prior, intercept_prior]).T
    ndims = initial_prior.shape[1]

    args = (y_data, dy_data, x_data)
    sampler = emcee.EnsembleSampler(walkers, ndims, comparision, args=args)

    sampler.run_mcmc(initial_prior, n_samples + n_burn, progress=progress)

    post_samples = sampler.get_chain(discard=n_burn).reshape((-1, ndims))

    return post_samples
