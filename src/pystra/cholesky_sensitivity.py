"""Differentiation of the Cholesky decomposition.

Implements the algorithm from the Appendix of Bourinet (2017) for
simultaneously computing the Cholesky factor L₀ and its derivative
∂L₀/∂θ from the modified correlation matrix R₀ and its derivative
∂R₀/∂θ.
"""

import numpy as np


def cholesky_with_derivative(R0, dR0):
    r"""Differentiate the Cholesky decomposition of R₀.

    Given a symmetric positive-definite matrix ``R0`` and the matrix
    ``dR0 = ∂R₀/∂θ`` (derivative w.r.t. some scalar parameter θ),
    simultaneously compute:

    - ``L0``  — the lower-triangular Cholesky factor such that
      ``R0 = L0 @ L0.T``
    - ``dL0`` — the lower-triangular matrix ``∂L₀/∂θ``

    This is a step-by-step differentiation of the standard Cholesky
    algorithm, following the Appendix of Bourinet (2017).

    Parameters
    ----------
    R0 : ndarray, shape (n, n)
        Symmetric positive-definite matrix (the modified Nataf
        correlation matrix).
    dR0 : ndarray, shape (n, n)
        Derivative ``∂R₀/∂θ``.  Must be symmetric.

    Returns
    -------
    L0 : ndarray, shape (n, n)
        Lower-triangular Cholesky factor.
    dL0 : ndarray, shape (n, n)
        Lower-triangular derivative ``∂L₀/∂θ``.
    """
    n = R0.shape[0]

    # Work on copies so the originals are not modified
    A = R0.copy().astype(float)
    dA = dR0.copy().astype(float)

    for k in range(n):
        # Lines 5-6: differentiate a_kk = sqrt(a_kk)
        dA[k, k] = dA[k, k] / (2 * np.sqrt(A[k, k]))
        A[k, k] = np.sqrt(A[k, k])

        # Lines 7-9: differentiate a_ik = a_ik / a_kk  for i > k
        for i in range(k + 1, n):
            dA[i, k] = (dA[i, k] * A[k, k] - A[i, k] * dA[k, k]) / A[k, k] ** 2
            A[i, k] = A[i, k] / A[k, k]

        # Lines 11-15: differentiate a_ij = a_ij - a_ik * a_jk  for j > k, i >= j
        for j in range(k + 1, n):
            for i in range(j, n):
                dA[i, j] = dA[i, j] - dA[i, k] * A[j, k] - A[i, k] * dA[j, k]
                A[i, j] = A[i, j] - A[i, k] * A[j, k]

    # Lines 18-19: extract lower triangular parts
    L0 = np.tril(A)
    dL0 = np.tril(dA)

    return L0, dL0


def dinvL0_dtheta(L0, dL0):
    r"""Derivative of the inverse Cholesky factor (Bourinet Eq. 19).

    .. math::
        \frac{\partial \mathbf{L}_0^{-1}}{\partial\theta}
        = -\mathbf{L}_0^{-1}\,
          \frac{\partial\mathbf{L}_0}{\partial\theta}\,
          \mathbf{L}_0^{-1}

    Parameters
    ----------
    L0 : ndarray, shape (n, n)
        Lower-triangular Cholesky factor.
    dL0 : ndarray, shape (n, n)
        Derivative ``∂L₀/∂θ`` (lower-triangular).

    Returns
    -------
    ndarray, shape (n, n)
        Derivative ``∂L₀⁻¹/∂θ``.
    """
    L0_inv = np.linalg.inv(L0)
    return -L0_inv @ dL0 @ L0_inv
