#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""Numerical integration for the Nataf modified correlation matrix.

The Nataf model requires solving for a modified correlation coefficient
``rho_0`` in standard-normal space that reproduces a given correlation
coefficient ``rho`` in physical space.  The functions in this module
evaluate the double integral that relates ``rho_0`` to the physical
correlation using 2-D Gauss-Legendre quadrature.
"""

from .quadrature import quadratureRule

import numpy as np


def rho_integral(rho0, margi, margj, Z1, Z2, X1, X2, WIP, detJ):
    r"""Evaluate the Nataf correlation integral for a trial ``rho_0``.

    Computes the physical-space correlation that results from a
    standard-normal-space correlation of ``rho0``, by numerically
    integrating the bivariate standard normal density weighted by the
    normalised marginal values.

    Parameters
    ----------
    rho0 : float
        Trial correlation in standard-normal space.
    margi, margj : Distribution
        Marginal distributions of the two random variables.
    Z1, Z2 : ndarray
        Meshgrid arrays of standard-normal integration coordinates.
    X1, X2 : ndarray
        Corresponding physical-space values via the marginal
        transformations.
    WIP : ndarray
        Outer product of quadrature weights.
    detJ : float
        Jacobian determinant of the coordinate mapping from
        ``[-zmax, zmax]`` to ``[-1, 1]``.

    Returns
    -------
    float
        The resulting physical-space correlation coefficient.
    """
    PHI2 = (
        1
        * (2 * np.pi * np.sqrt(1 - rho0**2)) ** (-1)
        * np.exp(
            -1
            * (2 * (1 - rho0**2)) ** (-1)
            * (Z1**2 - 2 * rho0 * Z1 * Z2 + Z2**2)
        )
    )
    rho = np.sum(
        np.sum(
            ((X1 - margi.mean) / margi.stdv)
            * ((X2 - margj.mean) / margj.stdv)
            * PHI2
            * detJ
            * WIP
        )
    )
    return rho


def zi_and_xi(margi, margj, zmax, nIP):
    """Set up the 2-D quadrature grid for the Nataf correlation integral.

    Computes the meshgrid arrays of standard-normal coordinates (Z1, Z2),
    their physical-space counterparts (X1, X2), the outer-product weight
    matrix, and the Jacobian determinant for the coordinate mapping.

    Parameters
    ----------
    margi, margj : Distribution
        Marginal distributions of the two random variables.
    zmax : float
        Truncation limit for the integration domain
        (``[-zmax, zmax]`` on each axis).
    nIP : int
        Number of Gauss-Legendre integration points per axis.

    Returns
    -------
    Z1, Z2 : ndarray
        Meshgrid arrays of standard-normal coordinates, each of
        shape ``(nIP, nIP)``.
    X1, X2 : ndarray
        Physical-space values corresponding to Z1 and Z2.
    WIP : ndarray
        Outer product of quadrature weights, shape ``(nIP, nIP)``.
    detJ : float
        Jacobian determinant of the affine coordinate mapping.
    """

    # Integration limits ( should be -infinity, +infinity on both axes, in theory )
    zmin = -zmax

    # Determinant of the jacobian of the transformation between
    # [z1max,z1min]x[z2max,z2min] and [-1,1]x[-1,1]
    detJ = (zmax - zmin) ** 2 * 4 ** (-1)

    # Get integration points and weight in [-1,1], nIP is the number of integration pts
    xIP, wIP = quadratureRule(nIP)

    # Transform integration points coordinates from [-1,1] to [zmax,zmin]
    z1 = zmin * np.ones(len(xIP)) + (zmax - zmin) * (xIP + np.ones(len(xIP))) / 2
    z2 = z1

    x1 = margi.u_to_x(z1)
    x2 = margj.u_to_x(z2)

    v1 = np.ones(nIP)
    v2 = np.transpose([v1])

    Z1 = np.dot(np.transpose([z1]), [v1])
    Z2 = np.dot(v2, [z2])
    X1 = np.dot(np.transpose([x1]), [v1])
    X2 = np.dot(v2, [x2])
    WIP = np.dot(np.transpose([wIP]), [wIP])

    return Z1, Z2, X1, X2, WIP, detJ


def _phi2(Z1, Z2, rho0):
    r"""Bivariate standard normal density.

    Parameters
    ----------
    Z1, Z2 : ndarray
        Standard-normal coordinate meshgrids.
    rho0 : float
        Correlation coefficient.

    Returns
    -------
    ndarray
        Density values, same shape as Z1 and Z2.
    """
    c = 1 - rho0**2
    Q = Z1**2 - 2 * rho0 * Z1 * Z2 + Z2**2
    return (2 * np.pi * np.sqrt(c)) ** (-1) * np.exp(-Q / (2 * c))


def _dphi2_drho0(Z1, Z2, rho0, PHI2):
    r"""Derivative of the bivariate standard normal density w.r.t. ``rho0``.

    Parameters
    ----------
    Z1, Z2 : ndarray
        Standard-normal coordinate meshgrids.
    rho0 : float
        Correlation coefficient.
    PHI2 : ndarray
        Pre-computed bivariate density (from :func:`_phi2`).

    Returns
    -------
    ndarray
        :math:`\partial\varphi_2 / \partial\rho_0`, same shape as Z1.
    """
    c = 1 - rho0**2
    numer = rho0 * c + Z1 * Z2 * (1 + rho0**2) - rho0 * (Z1**2 + Z2**2)
    return PHI2 * numer / c**2


def drho_drho0(rho0, margi, margj, Z1, Z2, X1, X2, WIP, detJ):
    r"""Derivative of the Nataf integral w.r.t. ``rho0`` (Bourinet Eq. 21).

    .. math::
        \frac{\partial\rho_{ij}}{\partial\rho_{0,ij}}
        = \int\!\!\int h\,
          \frac{\partial\varphi_2}{\partial\rho_{0,ij}}\,
          \mathrm{d}z_i\,\mathrm{d}z_j

    Parameters
    ----------
    rho0 : float
        Modified correlation in standard-normal space.
    margi, margj : Distribution
        Marginal distributions.
    Z1, Z2, X1, X2, WIP, detJ
        Quadrature grid from :func:`zi_and_xi`.

    Returns
    -------
    float
        :math:`\partial\rho_{ij}/\partial\rho_{0,ij}`.
    """
    PHI2 = _phi2(Z1, Z2, rho0)
    dPHI2 = _dphi2_drho0(Z1, Z2, rho0, PHI2)

    H = ((X1 - margi.mean) / margi.stdv) * ((X2 - margj.mean) / margj.stdv)
    return np.sum(H * dPHI2 * detJ * WIP)


def drho0_dtheta(rho0, margi, margj, Z1, Z2, X1, X2, WIP, detJ, var_idx, param):
    r"""Sensitivity of the modified correlation to a marginal parameter.

    Solves Eq. (23) of Bourinet (2017) for
    :math:`\partial\rho_{0,ij}/\partial\theta_k` by setting
    :math:`\partial\rho_{ij}/\partial\theta_k = 0` (physical correlation
    is fixed) and using Eq. (21) for the denominator:

    .. math::
        \frac{\partial\rho_{0,ij}}{\partial\theta_k}
        = -\frac{\displaystyle\int\!\!\int
            \frac{\partial h}{\partial\theta_k}\,\varphi_2\,
            \mathrm{d}z_i\,\mathrm{d}z_j}
           {\displaystyle\frac{\partial\rho_{ij}}
            {\partial\rho_{0,ij}}}

    The derivative :math:`\partial h/\partial\theta_k` uses the general
    formula derived from :math:`h = (X - \mu)/\sigma`:

    .. math::
        \frac{\partial h}{\partial\theta_k}
        = \frac{1}{\sigma}\!\left(
            \frac{\partial X}{\partial\theta_k}
          - \frac{\partial\mu}{\partial\theta_k}\right)
        - \frac{h}{\sigma}\,
          \frac{\partial\sigma}{\partial\theta_k}

    For ``"mean"`` this reduces to :math:`(\partial X/\partial\mu - 1)/\sigma`
    and for ``"std"`` to :math:`\partial X/\partial\sigma / \sigma - h/\sigma`.
    For any other parameter (e.g. a shape parameter), the moment
    derivatives :math:`\partial\mu/\partial\theta` and
    :math:`\partial\sigma/\partial\theta` are obtained from
    :meth:`Distribution._dmoments_dtheta`.

    Parameters
    ----------
    rho0 : float
        Modified correlation coefficient for this pair.
    margi, margj : Distribution
        Marginal distributions of variables *i* and *j*.
    Z1, Z2, X1, X2, WIP, detJ
        Quadrature grid from :func:`zi_and_xi`.
    var_idx : {0, 1}
        Which variable the parameter belongs to (0 → *margi*, 1 → *margj*).
    param : str
        Parameter name — any key from the distribution's
        :attr:`~Distribution.sensitivity_params` (e.g. ``"mean"``,
        ``"std"``, ``"shape"``).

    Returns
    -------
    float
        :math:`\partial\rho_{0,ij}/\partial\theta_k`.
    """
    PHI2 = _phi2(Z1, Z2, rho0)

    H_i = (X1 - margi.mean) / margi.stdv
    H_j = (X2 - margj.mean) / margj.stdv

    if var_idx == 0:
        dist, X, H_this, H_other = margi, X1, H_i, H_j
    else:
        dist, X, H_this, H_other = margj, X2, H_j, H_i

    sigma = dist.stdv

    # ∂X/∂θ_k = -(∂F(X)/∂θ_k) / f(X)
    dF = dist.dF_dtheta(X)
    fX = dist.pdf(X)
    # Clip fX away from zero to avoid division by zero in the tails
    fX = np.maximum(fX, 1e-300)
    dX = -dF[param] / fX

    # General ∂h/∂θ formula: (∂X/∂θ − ∂μ/∂θ)/σ − h·(∂σ/∂θ)/σ
    dmu, dsig = dist._dmoments_dtheta(param)
    dh_this = (dX - dmu) / sigma - H_this * dsig / sigma

    dh = dh_this * H_other

    # Numerator: ∫∫ (∂h/∂θ_k) × φ₂ dz_i dz_j
    numer = np.sum(dh * PHI2 * detJ * WIP)

    # Denominator: ∂ρ_ij/∂ρ₀,ij (Eq. 21)
    dPHI2 = _dphi2_drho0(Z1, Z2, rho0, PHI2)
    H = H_i * H_j
    denom = np.sum(H * dPHI2 * detJ * WIP)

    return -numer / denom
