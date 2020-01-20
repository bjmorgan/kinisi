"""
Simple utility functions 

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