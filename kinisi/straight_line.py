"""
MCMC evaluation for straightline.

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

import numpy as np


def prior(initial_guess, size):
    """
    Generate initialisation guess

    Args:
        initial_guess (float): Initial value to generate prior from.
        size (int): Number of walkers from MCMC.

    Returns:
        (array_like): Uniform prior centred on initial_guess.
    """
    minimum = initial_guess - 10. * initial_guess
    maximum = initial_guess + 10. * initial_guess

    return np.random.uniform(minimum, maximum, size)
