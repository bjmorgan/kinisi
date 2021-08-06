"""
A small module to find the nearest positive definite matrix.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import warnings
import numpy as np


def find_nearest_positive_definite(matrix):
    """
    Find the nearest positive-definite matrix to that given, using the method from
    N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): 10.1016/0024-3795(88)90223-6

    Args:
        matrix (:py:attr:`array_like`): Matrix to find nearest positive-definite for.
    Returns:
        :py:attr:`array_like`: Nearest positive-definite matrix.
    """

    if check_positive_definite(matrix):
        return matrix

    warnings.warn("The estimated covariance matrix was not positive definite, the nearest positive definite matrix has been found and will be used.")

    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if check_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(matrix))
    eye = np.eye(matrix.shape[0])
    k = 1
    while not check_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += eye * (-mineig * k**2 + spacing)
        k += 1

    return A3


def check_positive_definite(matrix):
    """
    Checks if a matrix is positive-definite via Cholesky decomposition.

    Args:
        matrix (:py:attr:`array_like`): Matrix to check.
    Returns:
        :py_attr:`bool`: True for a positive-definite matrix.


    Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
