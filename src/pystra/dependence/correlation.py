"""Correlation matrix handling and Nataf correlation modification.

The Nataf model transforms correlated non-normal random variables into
correlated standard-normal variables.  When the marginal distributions
are non-normal, the correlation matrix in standard normal space (the
*modified* correlation matrix ``Ro``) differs from the physical-space
correlation matrix ``R``.  This module provides the numerical
procedure that finds ``Ro`` from ``R`` and the marginal distributions.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import scipy.optimize as opt

import pystra as _pystra

from ..errors import ModelError

from .._numerics.integration import zi_and_xi, rho_integral

__all__ = [
    "CorrelationMatrix",
    "compute_modified_correlation_matrix",
    "absolute_integral_value",
    "set_modified_correlation_matrix",
]


def _validated(matrix):
    """Return *matrix* as a read-only array, if it is a valid correlation matrix."""
    try:
        R = np.array(matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise ModelError("A correlation matrix must be numeric") from error
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] == 0:
        raise ModelError("A correlation matrix must be square")
    if not np.all(np.isfinite(R)):
        raise ModelError("A correlation matrix must be finite")
    if not np.allclose(R, R.T, rtol=0, atol=1e-10):
        raise ModelError("A correlation matrix must be symmetric")
    if not np.allclose(np.diag(R), 1, rtol=0, atol=1e-10):
        raise ModelError("A correlation matrix must have a unit diagonal")
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ModelError(
            "A correlation matrix must be positive definite; "
            "see CorrelationMatrix.nearest_positive_definite"
        ) from None
    R.flags.writeable = False
    return R


class CorrelationMatrix:
    r"""Validated correlation matrix of the physical variables.

    The :math:`(i, j)` entry is :math:`\text{corr}(X_i, X_j)`. The matrix is
    checked when it is created: square, finite, symmetric, with a unit
    diagonal and positive definite. It is held as a read-only array, so build
    a new matrix rather than editing one. Elements can be read with ``[]``,
    and ``numpy.asarray`` returns the matrix.

    Parameters
    ----------
    matrix : array_like
        Symmetric positive-definite matrix with a unit diagonal.

    Raises
    ------
    ModelError
        If the matrix is not a valid correlation matrix.
    """

    def __init__(self, matrix: ArrayLike) -> None:
        self._matrix = _validated(matrix)

    @property
    def matrix(self) -> np.ndarray:
        """The matrix, as a read-only array."""
        return self._matrix

    def __repr__(self):
        return f"CorrelationMatrix({self._matrix.tolist()!r})"

    def __getitem__(self, key):
        return self._matrix[key]

    def __array__(self, dtype=None, copy=None):
        return np.array(self._matrix, dtype=dtype)

    def get_matrix(self) -> np.ndarray:
        """Return the correlation matrix as a read-only NumPy array."""
        return self._matrix

    def cholesky(self) -> np.ndarray:
        """Return the lower-triangular factor :math:`L` with :math:`R = L L^T`."""
        return np.linalg.cholesky(self._matrix)

    def nataf(self, model: "_pystra.StochasticModel") -> "CorrelationMatrix":
        """Return the Nataf correlation for this matrix and the model's marginals.

        The correlation of the standard-normal variables that the Nataf
        transformation uses to reproduce this correlation between physical
        variables with the marginal distributions of *model*. For normal
        marginals it equals this matrix.

        Parameters
        ----------
        model : StochasticModel
            Supplies the marginal distributions, in matrix order.

        Returns
        -------
        CorrelationMatrix
        """
        marginals = model.get_marginal_distributions()
        n = len(self._matrix)
        if len(marginals) != n:
            raise ModelError(
                f"The model has {len(marginals)} random variables; the matrix is {n} x {n}"
            )
        try:
            return CorrelationMatrix(_nataf_correlation(marginals, self._matrix))
        except ModelError as error:
            raise ModelError(
                f"The Nataf correlation for these marginals is invalid: {error}"
            ) from None

    @classmethod
    def nearest_positive_definite(
        cls,
        matrix: ArrayLike,
        *,
        min_eigenvalue: float = 1e-10,
        max_iterations: int = 100,
    ) -> "CorrelationMatrix":
        """Return the nearest valid correlation matrix to *matrix*.

        Uses Higham's (2002) alternating projections between the positive
        semidefinite matrices and the matrices with a unit diagonal, then
        raises the eigenvalues to at least ``min_eigenvalue`` so that the
        result is positive definite, and restores the unit diagonal. Use it to
        repair an estimate, for example one assembled from pairwise data, that
        is not positive definite.

        Parameters
        ----------
        matrix : array_like
            Square matrix, symmetrized before projection.
        min_eigenvalue : float, default 1e-10
            Smallest eigenvalue of the result before the diagonal is restored.
        max_iterations : int, default 100
            Limit on the alternating projections.

        Returns
        -------
        CorrelationMatrix

        References
        ----------
        Higham, N. J. (2002). Computing the nearest correlation matrix: a
        problem from finance. *IMA Journal of Numerical Analysis*, 22(3),
        329-343.
        """
        A = np.array(matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.all(np.isfinite(A)):
            raise ModelError(
                "nearest_positive_definite requires a finite square matrix"
            )
        if not np.isfinite(min_eigenvalue) or min_eigenvalue <= 0:
            raise ModelError("min_eigenvalue must be finite and positive")
        if (
            isinstance(max_iterations, bool)
            or int(max_iterations) != max_iterations
            or max_iterations < 1
        ):
            raise ModelError("max_iterations must be a positive integer")
        Y = (A + A.T) / 2
        correction = np.zeros_like(Y)
        for _ in range(int(max_iterations)):
            R = Y - correction
            values, vectors = np.linalg.eigh(R)
            X = (vectors * np.maximum(values, 0)) @ vectors.T
            correction = X - R
            previous, Y = Y, X.copy()
            np.fill_diagonal(Y, 1.0)
            if np.linalg.norm(Y - previous) <= 1e-12 * max(
                1.0, np.linalg.norm(previous)
            ):
                break
        values, vectors = np.linalg.eigh((Y + Y.T) / 2)
        X = (vectors * np.maximum(values, min_eigenvalue)) @ vectors.T
        scale = np.sqrt(np.diag(X))
        X = X / np.outer(scale, scale)
        X = (X + X.T) / 2
        np.fill_diagonal(X, 1.0)
        return cls(X)


def compute_modified_correlation_matrix(
    stochastic_model: "_pystra.StochasticModel",
) -> np.ndarray:
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
        ``(n, n)`` in standard normal space.
    """
    copula = stochastic_model.get_copula()
    if copula is not None:
        from .copula import GaussianCopula, StudentTCopula

        if isinstance(copula, GaussianCopula) and not isinstance(
            copula, StudentTCopula
        ):
            return copula.correlation
        raise ValueError(
            "Gaussian correlation modification does not apply to this copula"
        )
    return _nataf_correlation(
        stochastic_model.get_marginal_distributions(),
        stochastic_model.get_correlation(),
    )


def _nataf_correlation(marg, R):
    """Solve the Nataf integral equation for each correlated pair of marginals."""
    nvr = len(marg)
    n, m = np.shape(R)
    # copy() ensures the array is writable; np.eye may return a read-only
    # array in NumPy 2.0+.
    Ro = np.eye(n, m).copy()
    for i in range(nvr):
        for j in range(i):
            rho = R[i][j]
            if rho != 0:
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

                par = opt.fmin(
                    absolute_integral_value,
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


def absolute_integral_value(rho0: float | np.ndarray, *args: Any) -> float:
    r"""Objective function for the Nataf correlation optimization.

    Returns ``|rho_target - rho_integral(rho0)|``, which is minimized
    by ``scipy.optimize.fmin`` to find the modified correlation
    coefficient ``rho0`` in standard normal space.

    Parameters
    ----------
    rho0 : float
        Trial correlation in standard normal space.
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


def set_modified_correlation_matrix(
    stochastic_model: "_pystra.StochasticModel",
) -> None:
    """Compute the modified correlation matrix and store it on the model.

    Convenience wrapper that calls
    :func:`compute_modified_correlation_matrix` and assigns the result to
    the stochastic model via ``set_modified_correlation``.

    Parameters
    ----------
    stochastic_model : StochasticModel
        The model whose modified correlation matrix will be set.
    """

    Ro = compute_modified_correlation_matrix(stochastic_model)
    stochastic_model.set_modified_correlation(Ro)
