"""Tests for the Transformation class."""

import pytest
import numpy as np
from pystra.transformation import Transformation
from pystra.distributions import Normal, Lognormal, Uniform


class TestTransformationInit:
    def test_default_is_cholesky(self):
        t = Transformation()
        assert t.transform_type == "cholesky"

    def test_explicit_cholesky(self):
        t = Transformation("cholesky")
        assert t.transform_type == "cholesky"

    def test_explicit_svd(self):
        t = Transformation("svd")
        assert t.transform_type == "svd"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Undefined transformation type"):
            Transformation("invalid")


class TestCholeskyTransform:
    def test_uncorrelated_roundtrip(self):
        """u → x → u should round-trip for uncorrelated normals."""
        marg = [Normal("X1", 10, 2), Normal("X2", 5, 1)]
        Ro = np.eye(2)

        t = Transformation("cholesky")
        t.compute(Ro)

        for u_test in [np.array([0.0, 0.0]), np.array([1.0, -1.0]), np.array([-2.0, 2.0])]:
            x = t.u_to_x(u_test, marg)
            u_back = t.x_to_u(x, marg)
            np.testing.assert_allclose(u_back, u_test, atol=1e-6)

    def test_correlated_roundtrip(self):
        """u → x → u should round-trip for correlated normals."""
        marg = [Normal("X1", 10, 2), Normal("X2", 5, 1)]
        Ro = np.array([[1.0, 0.5], [0.5, 1.0]])

        t = Transformation("cholesky")
        t.compute(Ro)

        for u_test in [np.array([0.0, 0.0]), np.array([1.5, -0.5])]:
            x = t.u_to_x(u_test, marg)
            u_back = t.x_to_u(x, marg)
            np.testing.assert_allclose(u_back, u_test, atol=1e-6)

    def test_jacobian_shape(self):
        """Jacobian should have shape (nrv, nrv)."""
        marg = [Normal("X1", 10, 2), Normal("X2", 5, 1)]
        Ro = np.eye(2)

        t = Transformation("cholesky")
        t.compute(Ro)

        u = np.array([0.0, 0.0])
        x = t.u_to_x(u, marg)
        J = t.jacobian(u, x, marg)
        assert J.shape == (2, 2)

    def test_jacobian_uncorrelated_normals(self):
        """For uncorrelated normals, Jacobian diagonal = 1/stdv."""
        marg = [Normal("X1", 10, 2), Normal("X2", 5, 3)]
        Ro = np.eye(2)

        t = Transformation("cholesky")
        t.compute(Ro)

        u = np.array([0.0, 0.0])
        x = t.u_to_x(u, marg)
        J = t.jacobian(u, x, marg)
        assert pytest.approx(J[0, 0], abs=1e-6) == 1 / 2
        assert pytest.approx(J[1, 1], abs=1e-6) == 1 / 3

    def test_T_and_inv_T_are_set(self):
        """After compute, T and inv_T should be defined."""
        Ro = np.eye(2)
        t = Transformation("cholesky")
        t.compute(Ro)
        assert t.T is not None
        assert t.inv_T is not None

    def test_T_times_inv_T_is_identity(self):
        """T @ inv_T should be identity."""
        Ro = np.array([[1.0, 0.3], [0.3, 1.0]])
        t = Transformation("cholesky")
        t.compute(Ro)
        product = t.T @ t.inv_T
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)


class TestSVDTransform:
    def test_uncorrelated_roundtrip(self):
        """u → x → u should round-trip for uncorrelated normals."""
        marg = [Normal("X1", 10, 2), Normal("X2", 5, 1)]
        Ro = np.eye(2)

        t = Transformation("svd")
        t.compute(Ro)

        for u_test in [np.array([0.0, 0.0]), np.array([1.0, -1.0])]:
            x = t.u_to_x(u_test, marg)
            u_back = t.x_to_u(x, marg)
            np.testing.assert_allclose(u_back, u_test, atol=1e-6)

    def test_correlated_roundtrip(self):
        """u → x → u should round-trip for correlated normals."""
        marg = [Normal("X1", 10, 2), Normal("X2", 5, 1)]
        Ro = np.array([[1.0, 0.5], [0.5, 1.0]])

        t = Transformation("svd")
        t.compute(Ro)

        for u_test in [np.array([0.0, 0.0]), np.array([1.5, -0.5])]:
            x = t.u_to_x(u_test, marg)
            u_back = t.x_to_u(x, marg)
            np.testing.assert_allclose(u_back, u_test, atol=1e-6)

    def test_T_times_inv_T_is_identity(self):
        """T @ inv_T should be identity."""
        Ro = np.array([[1.0, 0.3], [0.3, 1.0]])
        t = Transformation("svd")
        t.compute(Ro)
        product = t.T @ t.inv_T
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)

    def test_cholesky_and_svd_both_roundtrip(self):
        """Both decompositions should independently round-trip correctly."""
        marg = [Normal("X1", 10, 2), Normal("X2", 5, 1)]
        Ro = np.array([[1.0, 0.4], [0.4, 1.0]])

        t_chol = Transformation("cholesky")
        t_chol.compute(Ro)

        t_svd = Transformation("svd")
        t_svd.compute(Ro)

        u = np.array([1.0, -0.5])

        # Both should round-trip independently
        x_chol = t_chol.u_to_x(u, marg)
        u_chol_back = t_chol.x_to_u(x_chol, marg)
        np.testing.assert_allclose(u_chol_back, u, atol=1e-6)

        x_svd = t_svd.u_to_x(u, marg)
        u_svd_back = t_svd.x_to_u(x_svd, marg)
        np.testing.assert_allclose(u_svd_back, u, atol=1e-6)


class TestTransformationWithMixedDistributions:
    def test_normal_lognormal_roundtrip(self):
        """Transform round-trip for Normal + Lognormal."""
        marg = [Normal("X1", 10, 2), Lognormal("X2", 5, 1)]
        Ro = np.eye(2)

        t = Transformation("cholesky")
        t.compute(Ro)

        u = np.array([0.5, -0.5])
        x = t.u_to_x(u, marg)
        u_back = t.x_to_u(x, marg)
        np.testing.assert_allclose(u_back, u, atol=1e-4)
