# -*- coding: utf-8 -*-

import numpy as np


class Transformation:
    """
    Nataf isoprobabilistic transformation between physical space (x) and
    standard normal space (u).

    The transformation relies on a square-root factorisation of the modified
    correlation matrix Ro, such that Ro = inv_T @ inv_T^T.  Two factorisations
    are available:

    - **Cholesky** (default): Ro = L @ L^T where L is lower-triangular.
    - **SVD**: Ro = (U sqrt(D)) @ (U sqrt(D))^T via the eigendecomposition of
      the symmetric positive-definite matrix Ro.

    Both factorisations satisfy the same identity and therefore produce
    identical reliability results (design point, reliability index beta,
    failure probability).  The intermediate correlated standard-normal vector
    z = inv_T @ u will in general differ between the two methods, but the
    final physical-space coordinates x are invariant because x_i = F_i^{-1}(
    Phi(z_i)) depends only on the marginal mapping.

    The SVD factorisation is generally more robust because it avoids computing
    the explicit inverse of a triangular factor; instead it works with the
    orthogonal eigenstructure of Ro.  It is recommended when Ro is
    near-singular or poorly conditioned.
    """

    def __init__(self, transform_type=None):
        """
        Initialization of the Transformation class
        """
        self.transform_types = ["cholesky", "svd"]

        self.transform_type = transform_type

        if self.transform_type is None:
            self.transform_type = "cholesky"

        if self.transform_type not in self.transform_types:
            raise ValueError("Undefined transformation type")

        self.T = None
        self.inv_T = None

    def x_to_u(self, x, marg):
        """
        Transformation from x (physical) to u (standard normal) space.

        Callers (e.g. FORM, SORM) may pass x as a 1-D array or as a column
        vector of shape (nrv, 1).  Flattening to 1-D with ``ravel()`` ensures
        that ``x[i]`` yields a scalar, which is what the marginal ``x_to_u``
        methods expect.
        """
        nrv = len(marg)
        x = np.asarray(x).ravel()
        u = np.zeros(nrv)
        for i in range(nrv):
            u[i] = marg[i].x_to_u(x[i])

        u = np.dot(self.T, u)
        return u

    def u_to_x(self, u, marg):
        """
        Transformation from u (standard normal) to x (physical) space.

        As with ``x_to_u``, the input is flattened to 1-D so that element
        indexing always produces a scalar for the marginal ``u_to_x`` calls.
        """
        nrv = len(marg)
        u = np.asarray(u).ravel()
        z = np.dot(self.inv_T, u)

        x = np.zeros(nrv)
        for i in range(nrv):
            x[i] = marg[i].u_to_x(z[i])
        return x

    def jacobian(self, u, x, marg):
        """
        Jacobian of the u-to-x transformation, J_{u,x}.

        Inputs are flattened to 1-D (see ``x_to_u``).  Each marginal
        ``jacobian(z_i, x_i)`` returns a diagonal ``np.ndarray`` via
        ``np.diag()``, even when called with scalar arguments — in that case
        the result is a (1, 1) matrix.  We wrap the scalar inputs with
        ``np.atleast_1d`` so that ``np.diag`` receives a 1-D array (it
        raises ``ValueError`` on 0-d input), and then extract the single
        element with ``.item()`` for assignment into the composite Jacobian.
        """
        nrv = len(marg)
        u = np.asarray(u).ravel()
        x = np.asarray(x).ravel()
        z = np.dot(self.inv_T, u)
        J_u_x = np.zeros((nrv, nrv))

        for i in range(nrv):
            Ji = marg[i].jacobian(np.atleast_1d(z[i]), np.atleast_1d(x[i]))
            J_u_x[i][i] = Ji.item()

        J_u_x = np.dot(self.T, J_u_x)
        return J_u_x

    def compute(self, Ro):
        """
        Compute the Isoprobabilistic Transformation using the chosen method
        """
        if self.transform_type == self.transform_types[0]:
            self._computeCholesky(Ro)
        elif self.transform_type == self.transform_types[1]:
            self._computeSVD(Ro)
        else:
            raise ValueError("Transform type not set")

    def _computeCholesky(self, Ro):
        """
        Compute Cholesky factorisation of the modified correlation matrix.

        Decomposes Ro = L @ L^T where L is lower-triangular, then sets::

            inv_T = L
            T     = L^{-1}

        This is the classical Nataf factorisation.  It requires Ro to be
        symmetric positive-definite (all eigenvalues strictly positive).
        """
        # Ro = self.model.getModifiedCorrelation()
        try:
            L = np.linalg.cholesky(Ro)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Cholesky decomposition failed — Ro may not be "
                f"positive-definite: {e}"
            ) from e

        self.T = np.linalg.inv(L)
        self.inv_T = L

    def _computeSVD(self, Ro):
        """
        Compute SVD-based factorisation of the modified correlation matrix.

        For the symmetric positive-definite matrix Ro the SVD coincides with
        the eigendecomposition: Ro = U @ diag(D) @ U^T.  The square-root
        factor is then R = U @ diag(sqrt(D)), giving::

            inv_T = R = U @ sqrt(D)
            T     = R^{-1}

        This satisfies the same identity as Cholesky (Ro = inv_T @ inv_T^T)
        but is more robust for ill-conditioned correlation matrices because
        the factorisation exploits the orthogonal eigenstructure rather than
        relying on triangular back-substitution.
        """
        try:
            U, D, V = np.linalg.svd(Ro)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"SVD failed: {e}"
            ) from e

        sqrtD = np.sqrt(D) * np.eye(len(D))
        R = U @ sqrtD

        self.T = np.linalg.inv(R)
        self.inv_T = R
