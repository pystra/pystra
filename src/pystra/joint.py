"""Marginals plus a copula, and their isoprobabilistic transformations."""

import numpy as np
from scipy.stats import norm, t

from .copula import Copula, GaussianCopula, StudentTCopula, _order
from .distributions import Distribution, ZeroInflated

__all__ = ["JointDistribution", "CopulaTransformation"]


class JointDistribution:
    """Continuous joint law specified by named Pystra marginals and a copula.

    Samples have rows of observations and columns in marginal order. The
    copula is a distribution specification, independent of the transformation
    later chosen for reliability analysis. Constants belong in StochasticModel.
    """

    def __init__(self, marginals, copula):
        self.marginals = tuple(marginals)
        if not isinstance(copula, Copula):
            raise TypeError("copula must implement the Copula interface")
        if len(self.marginals) != copula.dimension:
            raise ValueError("Copula dimension must match the marginal count")
        if any(not isinstance(m, Distribution) for m in self.marginals):
            raise TypeError("Marginals must be Pystra Distribution objects")
        if any(isinstance(m, ZeroInflated) and m.p > 0 for m in self.marginals):
            raise ValueError("Copula transformations require continuous marginals")
        self.names = tuple(m.get_name() for m in self.marginals)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Joint marginal names must be unique")
        self.copula = copula
        self.dimension = copula.dimension

    def _points(self, points):
        x = np.asarray(points, dtype=float)
        if x.ndim not in (1, 2) or x.shape[-1] != self.dimension or np.any(np.isnan(x)):
            raise ValueError("Expected points in joint marginal order")
        return x

    def cdf(self, points, **kwargs):
        x = self._points(points)
        p = np.stack([m.cdf(x[..., i]) for i, m in enumerate(self.marginals)], axis=-1)
        return self.copula.cdf(p, **kwargs)

    def logpdf(self, points):
        x = self._points(points)

        def one(row):
            p = np.array([m.cdf(row[i]) for i, m in enumerate(self.marginals)])
            densities = np.array([m.pdf(row[i]) for i, m in enumerate(self.marginals)])
            if np.any(densities == 0):
                return -np.inf
            if not np.all(np.isfinite(densities)) or np.any(densities < 0):
                raise ValueError(
                    "Joint density requires finite positive marginal densities"
                )
            return float(self.copula.logpdf(p) + np.log(densities).sum())

        return one(x) if x.ndim == 1 else np.array([one(row) for row in x])

    def pdf(self, points):
        return np.exp(self.logpdf(points))

    def rvs(self, size=1, seed=None):
        p = self.copula.rvs(size, seed)
        return np.column_stack([m.ppf(p[:, i]) for i, m in enumerate(self.marginals)])

    def make_transformation(
        self, method="rosenblatt", order=None, factorization="cholesky"
    ):
        """Build a normal Rosenblatt or spherical generalized Nataf mapping."""
        return CopulaTransformation(self, method, order, factorization)


class CopulaTransformation:
    """Isoprobabilistic map for an explicit JointDistribution.

    method='rosenblatt' maps to independent standard normals. method='nataf'
    requires an elliptical copula; Student-t produces a spherical t vector
    with identity *shape*, not independent coordinates or unit covariance.
    Its univariate scale is one; covariance is df/(df-2) times identity when
    df>2. Gaussian Nataf and canonical Gaussian Rosenblatt coincide.

    Order is a permutation of indices, supported by Rosenblatt only. Output
    coordinates remain indexed by original variables: u[order[k]] is the k-th
    conditional normal innovation. factorization='svd' is supported for Nataf.
    Jacobians are analytic for elliptical copulas, and use central differences
    of the inverse map in normal space for other copulas.
    """

    def __init__(
        self,
        joint_distribution,
        method="rosenblatt",
        order=None,
        factorization="cholesky",
    ):
        self.joint = joint_distribution
        self.copula = joint_distribution.copula
        self.marginals = joint_distribution.marginals
        self.dimension = joint_distribution.dimension
        if method not in ("rosenblatt", "nataf"):
            raise ValueError("method must be rosenblatt or nataf")
        if factorization not in ("cholesky", "svd"):
            raise ValueError("factorization must be cholesky or svd")
        if method == "nataf" and not self.copula.elliptical:
            raise ValueError("Generalized Nataf requires an elliptical copula")
        if method == "nataf" and order is not None:
            raise ValueError("Conditioning order applies only to Rosenblatt")
        if method == "rosenblatt" and factorization != "cholesky":
            raise ValueError(
                "Rosenblatt uses sequential conditioning, not an SVD factor"
            )
        self.method = method
        self.order = _order(order, self.dimension)
        self.transform_type = method
        self.is_student = isinstance(self.copula, StudentTCopula)
        self.standard_space = (
            "student_t" if self.is_student and method == "nataf" else "normal"
        )
        self.standard_marginal = (
            t(self.copula.df) if self.standard_space == "student_t" else norm
        )
        self.T = self.inv_T = None
        if self.copula.elliptical:
            R = self.copula.correlation[np.ix_(self.order, self.order)]
            if factorization == "cholesky":
                self.inv_T = np.linalg.cholesky(R)
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(R)
                self.inv_T = eigenvectors * np.sqrt(eigenvalues)
            self.T = np.linalg.inv(self.inv_T)

    def _vector(self, values):
        values = np.asarray(values, dtype=float).ravel()
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Transform requires a finite vector in full marginal order"
            )
        return values

    def _latent(self, x):
        if not self.is_student:
            return np.array([m.x_to_u(x[i]) for i, m in enumerate(self.marginals)])
        w = []
        for i, m in enumerate(self.marginals):
            p = float(m.cdf(x[i]))
            if p > 0.5 and m.dist_obj is not None:
                w.append(t.isf(m.dist_obj.sf(x[i]), self.copula.df))
            else:
                w.append(t.ppf(p, self.copula.df))
        return np.array(w)

    def _physical(self, w):
        if not self.is_student:
            return np.array([m.u_to_x(w[i]) for i, m in enumerate(self.marginals)])
        return np.array(
            [
                (
                    m.dist_obj.isf(t.sf(w[i], self.copula.df))
                    if w[i] > 0 and m.dist_obj is not None
                    else m.ppf(t.cdf(w[i], self.copula.df))
                )
                for i, m in enumerate(self.marginals)
            ]
        )

    def x_to_u(self, x, marg=None):
        x = self._vector(x)
        if not self.copula.elliptical:
            p = np.array([m.cdf(x[i]) for i, m in enumerate(self.marginals)])
            return self._vector(norm.ppf(self.copula.rosenblatt(p, self.order)))
        z = self.T @ self._latent(x)[self.order]
        result = np.empty(self.dimension)
        if self.is_student and self.method == "rosenblatt":
            for k, i in enumerate(self.order):
                q = z[k] / np.sqrt(
                    (self.copula.df + z[:k] @ z[:k]) / (self.copula.df + k)
                )
                result[i] = (
                    norm.isf(t.sf(q, self.copula.df + k))
                    if q > 0
                    else norm.ppf(t.cdf(q, self.copula.df + k))
                )
        else:
            result[self.order] = z
        return self._vector(result)

    def u_to_x(self, u, marg=None):
        u = self._vector(u)
        if not self.copula.elliptical:
            p = self.copula.inverse_rosenblatt(norm.cdf(u), self.order)
            return self._vector([m.ppf(p[i]) for i, m in enumerate(self.marginals)])
        z = u[self.order].copy()
        if self.is_student and self.method == "rosenblatt":
            for k, i in enumerate(self.order):
                q = (
                    t.isf(norm.sf(u[i]), self.copula.df + k)
                    if u[i] > 0
                    else t.ppf(norm.cdf(u[i]), self.copula.df + k)
                )
                z[k] = q * np.sqrt(
                    (self.copula.df + z[:k] @ z[:k]) / (self.copula.df + k)
                )
        w = np.empty(self.dimension)
        w[self.order] = self.inv_T @ z
        return self._vector(self._physical(w))

    def jacobian(self, u, x, marg=None):
        """Return du/dx in original variable order."""
        u, x = self._vector(u), self._vector(x)
        if not self.copula.elliptical:
            h = np.cbrt(np.finfo(float).eps) * (1 + np.abs(u))
            inverse = np.empty((self.dimension, self.dimension))
            for j in range(self.dimension):
                step = np.zeros(self.dimension)
                step[j] = h[j]
                inverse[:, j] = (self.u_to_x(u + step) - self.u_to_x(u - step)) / (
                    2 * h[j]
                )
            return np.linalg.inv(inverse)
        w = self._latent(x)[self.order]
        z = self.T @ w
        B = np.eye(self.dimension)
        if self.is_student and self.method == "rosenblatt":
            for k, i in enumerate(self.order):
                denom = self.copula.df + z[:k] @ z[:k]
                scale = np.sqrt(denom / (self.copula.df + k))
                q = z[k] / scale
                multiplier = np.exp(t.logpdf(q, self.copula.df + k) - norm.logpdf(u[i]))
                B[k, k] = multiplier / scale
                B[k, :k] = -multiplier * q * z[:k] / denom
        density = t.pdf(w, self.copula.df) if self.is_student else norm.pdf(w)
        physical = np.array([self.marginals[i].pdf(x[i]) for i in self.order])
        if np.any(density <= 0) or np.any(physical <= 0):
            raise ValueError("Transform Jacobian requires positive finite densities")
        ordered = B @ self.T @ np.diag(physical / density)
        result = np.empty_like(ordered)
        result[np.ix_(self.order, self.order)] = ordered
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite transformation Jacobian")
        return result
