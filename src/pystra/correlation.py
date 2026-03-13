#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""Correlation matrix handling and Nataf correlation modification.

The Nataf model transforms correlated non-normal random variables into
correlated standard-normal variables.  When the marginal distributions
are non-normal, the correlation matrix in standard-normal space (the
*modified* correlation matrix ``Ro``) differs from the physical-space
correlation matrix ``R``.  This module provides the numerical
procedure that finds ``Ro`` from ``R`` and the marginal distributions.
"""

import numpy as np
import scipy.optimize as opt

from .integration import zi_and_xi, rho_integral


class CorrelationMatrix:
    r"""Physical-space correlation matrix wrapper.

    A thin wrapper around a NumPy array that stores the correlation
    matrix of :math:`n` random variables :math:`X_1, \dots, X_n`.
    The :math:`(i, j)` entry is :math:`\text{corr}(X_i, X_j)`.

    Supports element access via ``[]`` so it can be used interchangeably
    with a plain NumPy array in most contexts.

    Parameters
    ----------
    matrix : array_like, optional
        Symmetric correlation matrix.  Must have ones on the diagonal
        and all eigenvalues positive (positive definite).
    """

    def __init__(self, matrix=None):
        self.matrix = matrix
        self.mu = None
        self.sigma = None
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p4 = None

    def __repr__(self):
        return repr(self.matrix)

    def __getitem__(self, key):
        return self.matrix[key]

    def __setitem__(self, key, item):
        self.matrix[key] = item

    def getMatrix(self):
        """Return the correlation matrix as a NumPy array.

        Returns
        -------
        ndarray
            The correlation matrix.
        """
        return self.matrix


def computeModifiedCorrelationMatrix(stochastic_model):
    r"""Compute the modified (Nataf) correlation matrix.

    For each pair of non-normal marginals, the physical-space
    correlation :math:`\rho_{ij}` is mapped to the standard-normal-space
    correlation :math:`\rho_{0,ij}` by numerically solving the
    bi-folded integral equation.  For jointly normal pairs the
    mapping is the identity.

    The number of quadrature points is increased adaptively for
    correlations close to :math:`\pm 1` to maintain accuracy.

    Parameters
    ----------
    stochastic_model : StochasticModel
        The stochastic model containing marginal distributions and the
        physical-space correlation matrix.

    Returns
    -------
    ndarray
        The symmetric modified correlation matrix ``Ro`` of shape
        ``(n, n)`` in standard-normal space.
    """
    marg = stochastic_model.getMarginalDistributions()
    R = stochastic_model.getCorrelation()
    nvr = len(marg)
    n, m = np.shape(R)
    # copy() ensures the array is writable; np.eye may return a read-only
    # array in NumPy 2.0+.
    Ro = np.eye(n, m).copy()
    flag_sens = True
    for i in range(nvr):
        for j in range(i):
            rho = R[i][j]
            if rho != 0 or flag_sens:
                margi = marg[i]
                margj = marg[j]

                zmax = 6

                if np.absolute(rho) > 0.9995:
                    nIP = 1024
                elif np.absolute(rho) > 0.998:
                    nIP = 512
                elif np.absolute(rho) > 0.992:
                    nIP = 256
                elif np.absolute(rho) > 0.97:
                    nIP = 128
                elif np.absolute(rho) > 0.9:
                    nIP = 64
                else:
                    nIP = 32

                Z1, Z2, X1, X2, WIP, detJ = zi_and_xi(margi, margj, zmax, nIP)

            if rho != 0:
                par = opt.fmin(
                    absoluteIntegralValue,
                    rho,
                    args=(rho, margi, margj, Z1, Z2, X1, X2, WIP, detJ),
                    disp=False,
                )
                rho0 = par[0]
            else:
                rho0 = 0

            Ro[i][j] = rho0

    Ro = Ro + np.transpose(np.tril(Ro, -1))

    # Some parts are missing !!!

    return Ro


def absoluteIntegralValue(rho0, *args):
    r"""Objective function for the Nataf correlation optimisation.

    Returns ``|rho_target - rho_integral(rho0)|``, which is minimised
    by ``scipy.optimize.fmin`` to find the modified correlation
    coefficient ``rho0`` in standard-normal space.

    Parameters
    ----------
    rho0 : float
        Trial correlation in standard-normal space.
    *args : tuple
        ``(rho_target, margi, margj, Z1, Z2, X1, X2, WIP, detJ)`` —
        the target physical-space correlation and the pre-computed
        quadrature grid (see :func:`zi_and_xi`).

    Returns
    -------
    float
        Absolute error between target and computed correlation.
    """
    rho_target, margi, margj, Z1, Z2, X1, X2, WIP, detJ = args

    f = np.absolute(
        rho_target - rho_integral(rho0, margi, margj, Z1, Z2, X1, X2, WIP, detJ)
    )
    return f


def setModifiedCorrelationMatrix(stochastic_model):
    """Compute the modified correlation matrix and store it on the model.

    Convenience wrapper that calls
    :func:`computeModifiedCorrelationMatrix` and assigns the result to
    the stochastic model via ``setModifiedCorrelation``.

    Parameters
    ----------
    stochastic_model : StochasticModel
        The model whose modified correlation matrix will be set.
    """

    Ro = computeModifiedCorrelationMatrix(stochastic_model)
    stochastic_model.setModifiedCorrelation(Ro)
