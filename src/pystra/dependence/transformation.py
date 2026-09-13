"""Normal Nataf coordinate transforms and directed Jacobians."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..distributions import Distribution

__all__ = ["Transformation"]


class Transformation:
    """Nataf isoprobabilistic transformation between physical space (x) and
    standard normal space (u).

    The transformation relies on a square-root factorization of the modified
    correlation matrix Ro, such that Ro = inv_T @ inv_T^T.  Two factorizations
    are available:

    - **Cholesky** (default): Ro = L @ L^T where L is lower-triangular.
    - **SVD**: Ro = (U sqrt(D)) @ (U sqrt(D))^T via the eigendecomposition of
      the symmetric positive-definite matrix Ro.

    The two independent-normal coordinate systems differ by an orthogonal
    transformation. Corresponding physical design points, beta and probability
    agree for the same optimum, up to numerical solver error. A fixed numerical
    u vector generally maps to different physical points under the two factors.

    The SVD factorization is generally more robust because it avoids computing
    the explicit inverse of a triangular factor; instead it works with the
    orthogonal eigenstructure of Ro.  It is recommended when Ro is
    near-singular or poorly conditioned.

    Parameters
    ----------
    transform_type : {"cholesky", "svd"}, optional
        Matrix factorization, defaulting to Cholesky. Call :meth:`compute`
        with the modified normal correlation before transforming a point."""

    standard_space = "normal"

    def __init__(self, transform_type: str | None = None) -> None:
        self.transform_types = ["cholesky", "svd"]

        self.transform_type = transform_type

        if self.transform_type is None:
            self.transform_type = "cholesky"

        if self.transform_type not in self.transform_types:
            raise ValueError("Undefined transformation type")

        self.T = None
        self.inv_T = None

    def x_to_u(self, x: ArrayLike, marg: Sequence[Distribution]) -> np.ndarray:
        """Map a physical point to independent standard normal coordinates.

        Parameters
        ----------
        x : array_like, shape (dimension,) or (dimension, 1)
            Physical point, flattened to one dimension before marginal transforms.
        marg : sequence of Distribution
            Marginals in model variable order, matching the computed correlation.

        Returns
        -------
        ndarray, shape (dimension,)
            Independent normal coordinates in the same variable order."""
        dimension = len(marg)
        x = np.asarray(x).ravel()
        u = np.zeros(dimension)
        for i in range(dimension):
            u[i] = marg[i].x_to_u(x[i])

        u = np.dot(self.T, u)
        return u

    def u_to_x(self, u: ArrayLike, marg: Sequence[Distribution]) -> np.ndarray:
        """Map an independent normal point to physical coordinates.

        Parameters
        ----------
        u : array_like, shape (dimension,) or (dimension, 1)
            Reference point, flattened to one dimension before transformation.
        marg : sequence of Distribution
            Marginals in model variable order, matching the computed correlation.

        Returns
        -------
        ndarray, shape (dimension,)
            Physical coordinates in marginal order."""
        dimension = len(marg)
        u = np.asarray(u).ravel()
        z = np.dot(self.inv_T, u)

        x = np.zeros(dimension)
        for i in range(dimension):
            x[i] = marg[i].u_to_x(z[i])
        return x

    def jacobian_u_wrt_x(
        self, u: ArrayLike, x: ArrayLike, marg: Sequence[Distribution]
    ) -> np.ndarray:
        """Return physical-to-reference derivatives ``du[i] / dx[j]``.

        Parameters
        ----------
        u, x : array_like
            Corresponding reference and physical points, shape ``(dimension,)``.
        marg : sequence of Distribution
            Marginals in model variable order.

        Returns
        -------
        ndarray
            Shape ``(dimension, dimension)``, with reference coordinates in
            rows and physical variables in columns, both in model order.
        """
        dimension = len(marg)
        u = np.asarray(u).ravel()
        x = np.asarray(x).ravel()
        z = np.dot(self.inv_T, u)
        J_u_x = np.zeros((dimension, dimension))

        for i in range(dimension):
            Ji = marg[i].jacobian(np.atleast_1d(z[i]), np.atleast_1d(x[i]))
            J_u_x[i][i] = Ji.item()

        J_u_x = np.dot(self.T, J_u_x)
        return J_u_x

    @property
    def dimension(self) -> int:
        """Number of coordinates; available after :meth:`compute`."""
        if self.T is None:
            raise ValueError("Compute the transformation before requesting dimension")
        return self.T.shape[0]

    def jacobian_x_wrt_u(
        self, u: ArrayLike, x: ArrayLike, marg: Sequence[Distribution]
    ) -> np.ndarray:
        """Return ``dx[i] / du[j]`` at corresponding reference/physical points.

        Inputs are vectors of shape ``(dimension,)`` in model variable order.
        The returned matrix has shape ``(dimension, dimension)`` with physical
        variables in rows and reference coordinates in columns. It is the
        inverse of :meth:`jacobian_u_wrt_x` at a nonsingular point.
        """
        return np.linalg.inv(self.jacobian_u_wrt_x(u, x, marg))

    def compute(self, Ro: ArrayLike) -> None:
        """Compute and store the selected correlation factors.

        Parameters
        ----------
        Ro : array_like, shape (dimension, dimension)
            Symmetric positive-definite modified normal correlation matrix.

        Raises
        ------
        numpy.linalg.LinAlgError
            If factorization or inversion fails."""
        if self.transform_type == self.transform_types[0]:
            self._compute_cholesky(Ro)
        elif self.transform_type == self.transform_types[1]:
            self._compute_svd(Ro)
        else:
            raise ValueError("Transform type not set")

    def _compute_cholesky(self, Ro):
        """
        Compute Cholesky factorization of the modified correlation matrix.

        Decomposes Ro = L @ L^T where L is lower-triangular, then sets::

            inv_T = L
            T     = L^{-1}

        This is the classical Nataf factorization.  It requires Ro to be
        symmetric positive-definite (all eigenvalues strictly positive).
        """
        try:
            L = np.linalg.cholesky(Ro)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Cholesky decomposition failed — Ro may not be "
                f"positive-definite: {e}"
            ) from e

        self.T = np.linalg.inv(L)
        self.inv_T = L

    def _compute_svd(self, Ro):
        """
        Compute SVD-based factorization of the modified correlation matrix.

        For the symmetric positive-definite matrix Ro the SVD coincides with
        the eigendecomposition: Ro = U @ diag(D) @ U^T.  The square-root
        factor is then R = U @ diag(sqrt(D)), giving::

            inv_T = R = U @ sqrt(D)
            T     = R^{-1}

        This satisfies the same identity as Cholesky (Ro = inv_T @ inv_T^T)
        but is more robust for ill-conditioned correlation matrices because
        the factorization exploits the orthogonal eigenstructure rather than
        relying on triangular back-substitution.
        """
        try:
            U, D, V = np.linalg.svd(Ro)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"SVD failed: {e}") from e

        sqrtD = np.sqrt(D) * np.eye(len(D))
        R = U @ sqrtD

        self.T = np.linalg.inv(R)
        self.inv_T = R
