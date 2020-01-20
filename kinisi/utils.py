"""
Simple utility functions

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""


def straight_line(abscissa, gradient, intercept):
    """
    Calculate ordinate of straight line

    args:
        abscissa (array_like): abscissa
        gradient (float): gradient
        intercept (float): intercept

    returns:
        array_like: ordinate
    """
    return gradient * abscissa + intercept
