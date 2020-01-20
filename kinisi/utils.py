"""
Simple utility functions 

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

def straight_line(x, m, c):
    """
    Calculate ordinate of straight line 

    args: 
        x (array_like): abscissa
        m (float): gradient
        c (float): intercept 

    returns: 
        array_like: ordinate
    """
    return m * x + c