"""Continuous copula specifications, independent of physical marginals."""

import numpy as np
from scipy.stats import norm, t, multivariate_normal, multivariate_t

__all__ = [
    "Copula",
    "GaussianCopula",
    "StudentTCopula",
    "FrankCopula",
    "IndependentCopula",
]


def _points(points, dimension, interior=False):
    points = np.asarray(points, dtype=float)
    if points.ndim not in (1, 2) or points.shape[-1] != dimension:
        raise ValueError(f"Expected a point or rows of points of dimension {dimension}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Copula coordinates must be finite")
    if interior:
        bad = (points <= 0) | (points >= 1)
    else:
        bad = (points < 0) | (points > 1)
    if np.any(bad):
        raise ValueError(
            "Copula probabilities outside the supported unit interval; "
            "transforms and densities require strictly interior probabilities"
        )
    return points


def _order(order, dimension):
    if order is None:
        return np.arange(dimension)
    values = np.asarray(order)
    if (
        values.shape != (dimension,)
        or values.dtype.kind not in "iu"
        or sorted(values.tolist()) != list(range(dimension))
    ):
        raise ValueError("order must be a permutation of variable indices")
    return values.astype(int)


def _sample_size(size):
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError("size must be a nonnegative integer")
    return int(size)


class Copula:
    """Interface for continuous dependence on the unit cube.

    Coordinates and returned samples follow the original marginal order.
    Subclasses provide cdf, logpdf, rosenblatt and inverse_rosenblatt. The
    Rosenblatt maps take/return uniform coordinates, not normal scores.
    """

    elliptical = False

    def pdf(self, probabilities):
        return np.exp(self.logpdf(probabilities))

    def rvs(self, size=1, seed=None):
        """Sample rows using a local NumPy random generator."""
        rng = np.random.default_rng(seed)
        # Exclude representable endpoints for inverse conditional transforms.
        q = rng.uniform(
            np.finfo(float).eps,
            1 - np.finfo(float).eps,
            (_sample_size(size), self.dimension),
        )
        return self.inverse_rosenblatt(q)


class GaussianCopula(Copula):
    """Gaussian copula specified by latent normal correlation R.

    R is not generally the physical Pearson correlation of the marginals.
    A finite, symmetric, positive-definite matrix with unit diagonal is required.
    """

    elliptical = True

    def __init__(self, correlation):
        matrix = np.array(correlation, dtype=float, copy=True)
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("Copula correlation must be a finite square matrix")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise ValueError("Copula correlation must be symmetric")
        if not np.allclose(np.diag(matrix), 1, rtol=0, atol=1e-12):
            raise ValueError("Copula correlation must have unit diagonal")
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1)
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("Copula correlation must be positive definite") from error
        self._correlation = matrix
        self.dimension = len(matrix)

    @property
    def correlation(self):
        """Copy of the latent correlation (shape matrix for Student-t)."""
        return self._correlation.copy()

    @property
    def kendall_tau(self):
        """Pairwise Kendall rank correlation implied by this elliptical copula."""
        return 2 / np.pi * np.arcsin(self._correlation)

    @classmethod
    def from_kendall_tau(cls, tau, **kwargs):
        """Construct from rank correlations, validating the resulting shape."""
        tau = np.asarray(tau, dtype=float)
        if not np.all(np.isfinite(tau)) or np.any(np.abs(tau) > 1):
            raise ValueError("Kendall tau must be finite and in [-1, 1]")
        return cls(np.sin(np.pi * tau / 2), **kwargs)

    def logpdf(self, probabilities):
        p = _points(probabilities, self.dimension, interior=True)
        z = norm.ppf(p)
        return multivariate_normal.logpdf(z, cov=self._correlation) - np.sum(
            norm.logpdf(z), axis=-1
        )

    def cdf(self, probabilities, **kwargs):
        """Evaluate the copula CDF; kwargs control SciPy integration."""
        p = _points(probabilities, self.dimension)

        def one(row):
            if np.any(row == 0):
                return 0.0
            active = np.flatnonzero(row < 1)
            if len(active) == 0:
                return 1.0
            if len(active) == 1:
                return float(row[active[0]])
            return float(
                multivariate_normal.cdf(
                    norm.ppf(row[active]),
                    cov=self._correlation[np.ix_(active, active)],
                    **kwargs,
                )
            )

        return one(p) if p.ndim == 1 else np.array([one(row) for row in p])

    def rosenblatt(self, probabilities, order=None):
        p = _points(probabilities, self.dimension, interior=True)
        ids = _order(order, self.dimension)
        L = np.linalg.cholesky(self._correlation[np.ix_(ids, ids)])
        z = np.linalg.solve(L, norm.ppf(p)[..., ids].T).T
        result = np.empty_like(p)
        result[..., ids] = norm.cdf(z)
        return _points(result, self.dimension, interior=True)

    def inverse_rosenblatt(self, probabilities, order=None):
        q = _points(probabilities, self.dimension, interior=True)
        ids = _order(order, self.dimension)
        L = np.linalg.cholesky(self._correlation[np.ix_(ids, ids)])
        result = np.empty_like(q)
        result[..., ids] = norm.cdf(norm.ppf(q)[..., ids] @ L.T)
        return _points(result, self.dimension, interior=True)


class IndependentCopula(GaussianCopula):
    """Product copula of the specified dimension."""

    def __init__(self, dimension):
        dimension = _sample_size(dimension)
        if dimension == 0:
            raise ValueError("dimension must be positive")
        super().__init__(np.eye(dimension))

    def cdf(self, probabilities, **kwargs):
        return np.prod(_points(probabilities, self.dimension), axis=-1)

    @classmethod
    def from_kendall_tau(cls, tau, **kwargs):
        matrix = GaussianCopula.from_kendall_tau(tau).correlation
        if kwargs or not np.array_equal(matrix, np.eye(len(matrix))):
            raise ValueError("IndependentCopula requires identity Kendall correlation")
        return cls(len(matrix))


class StudentTCopula(GaussianCopula):
    """Student-t copula with latent shape R and positive degrees of freedom.

    Identity R does not imply independence: coordinates share a random scale.
    The shape is not a physical Pearson correlation matrix. df need not exceed
    two, since arbitrary physical marginals can have finite moments regardless
    of the latent t representative's moments.
    """

    def __init__(self, correlation, df):
        super().__init__(correlation)
        if not np.isfinite(df) or df <= 0:
            raise ValueError("df must be finite and positive")
        self.df = float(df)

    def logpdf(self, probabilities):
        p = _points(probabilities, self.dimension, interior=True)
        w = t.ppf(p, self.df)
        return multivariate_t.logpdf(w, shape=self._correlation, df=self.df) - np.sum(
            t.logpdf(w, self.df), axis=-1
        )

    def cdf(self, probabilities, **kwargs):
        p = _points(probabilities, self.dimension)

        def one(row):
            if np.any(row == 0):
                return 0.0
            active = np.flatnonzero(row < 1)
            if len(active) == 0:
                return 1.0
            if len(active) == 1:
                return float(row[active[0]])
            return float(
                multivariate_t.cdf(
                    t.ppf(row[active], self.df),
                    df=self.df,
                    shape=self._correlation[np.ix_(active, active)],
                    **kwargs,
                )
            )

        return one(p) if p.ndim == 1 else np.array([one(row) for row in p])

    def rosenblatt(self, probabilities, order=None):
        p = _points(probabilities, self.dimension, interior=True)
        ids = _order(order, self.dimension)
        L = np.linalg.cholesky(self._correlation[np.ix_(ids, ids)])
        z = np.linalg.solve(L, t.ppf(p, self.df)[..., ids].T).T
        result = np.empty_like(p)
        for k, i in enumerate(ids):
            scale = np.sqrt(
                (self.df + np.sum(z[..., :k] ** 2, axis=-1)) / (self.df + k)
            )
            result[..., i] = t.cdf(z[..., k] / scale, self.df + k)
        return _points(result, self.dimension, interior=True)

    def inverse_rosenblatt(self, probabilities, order=None):
        q = _points(probabilities, self.dimension, interior=True)
        ids = _order(order, self.dimension)
        L = np.linalg.cholesky(self._correlation[np.ix_(ids, ids)])
        z = np.empty_like(q)
        for k, i in enumerate(ids):
            scale = np.sqrt(
                (self.df + np.sum(z[..., :k] ** 2, axis=-1)) / (self.df + k)
            )
            z[..., k] = scale * t.ppf(q[..., i], self.df + k)
        result = np.empty_like(q)
        result[..., ids] = t.cdf(z @ L.T, self.df)
        return _points(result, self.dimension, interior=True)


class FrankCopula(Copula):
    """Bivariate Frank copula; theta=0 gives independence.

    This initial implementation supports finite ``abs(theta) <= 30`` to keep the
    exponential conditional formulas resolved in double precision.
    """

    dimension = 2

    def __init__(self, theta):
        if not np.isfinite(theta) or abs(theta) > 30:
            raise ValueError("Frank theta must be finite with |theta| <= 30")
        self.theta = float(theta)

    def cdf(self, probabilities):
        p = _points(probabilities, 2)
        if self.theta == 0:
            return np.prod(p, axis=-1)
        # Radial symmetry avoids cancellation near the upper-right corner.
        reflected = np.sum(p, axis=-1) > 1
        offset = np.where(reflected, np.sum(p, axis=-1) - 1, 0)
        p = np.where(reflected[..., None], 1 - p, p)
        a = np.expm1(-self.theta * p)
        return (
            offset
            - np.log1p(a[..., 0] * (a[..., 1] / np.expm1(-self.theta))) / self.theta
        )

    def _denominator(self, p):
        if self.theta > 1:
            a = np.exp(-self.theta * p)
            return -(
                a[..., 0] + a[..., 1] - a[..., 0] * a[..., 1] - np.exp(-self.theta)
            )
        a = np.expm1(-self.theta * p)
        return np.expm1(-self.theta) + a[..., 0] * a[..., 1]

    def logpdf(self, probabilities):
        p = _points(probabilities, 2, interior=True)
        if self.theta == 0:
            return np.zeros(p.shape[:-1])
        th = self.theta
        D = np.expm1(-th)
        return (
            np.log(abs(th))
            + np.log(abs(D))
            - th * np.sum(p, axis=-1)
            - 2 * np.log(np.abs(self._denominator(p)))
        )

    def rosenblatt(self, probabilities, order=None):
        p = _points(probabilities, 2, interior=True)
        i, j = _order(order, 2)
        result = p.copy()
        if self.theta != 0:
            a = np.exp(-self.theta * p[..., i])
            b = np.expm1(-self.theta * p[..., j])
            result[..., j] = a * b / self._denominator(p)
        return _points(result, 2, interior=True)

    def inverse_rosenblatt(self, probabilities, order=None):
        q = _points(probabilities, 2, interior=True)
        i, j = _order(order, 2)
        result = q.copy()
        if self.theta != 0:
            a = np.exp(-self.theta * q[..., i])
            denominator = a * (1 - q[..., j]) + q[..., j]
            if abs(self.theta) > 1:
                numerator = a * (1 - q[..., j]) + q[..., j] * np.exp(-self.theta)
                result[..., j] = (np.log(denominator) - np.log(numerator)) / self.theta
            else:
                b = q[..., j] * np.expm1(-self.theta) / denominator
                result[..., j] = -np.log1p(b) / self.theta
        return _points(result, 2, interior=True)
