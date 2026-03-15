"""Tests for quadrature and integration modules."""

import pytest
import numpy as np
from pystra.quadrature import quadratureRule
from pystra.integration import zi_and_xi, rho_integral
from pystra.distributions import Normal


# ---------------------------------------------------------------------------
# Quadrature Rule
# ---------------------------------------------------------------------------


class TestQuadratureRule:
    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_points_in_range(self, n):
        """Gauss-Legendre points should lie in [-1, 1]."""
        bp, wf = quadratureRule(n)
        assert np.all(bp >= -1.0 - 1e-10)
        assert np.all(bp <= 1.0 + 1e-10)

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_weights_sum(self, n):
        """Gauss-Legendre weights should sum to 2 (integral of 1 over [-1,1])."""
        bp, wf = quadratureRule(n)
        assert pytest.approx(np.sum(wf), abs=1e-10) == 2.0

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_point_symmetry(self, n):
        """Points should be symmetric about 0."""
        bp, wf = quadratureRule(n)
        bp_sorted = np.sort(bp)
        assert pytest.approx(np.sum(bp_sorted), abs=1e-10) == 0.0

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_weight_symmetry(self, n):
        """Weights should be symmetric."""
        bp, wf = quadratureRule(n)
        # Sort by points, then weights should be symmetric
        idx = np.argsort(bp)
        wf_sorted = wf[idx]
        np.testing.assert_allclose(wf_sorted, wf_sorted[::-1], atol=1e-10)

    def test_correct_number_of_points(self):
        """Should return exactly n points and n weights."""
        for n in [2, 3, 5, 10]:
            bp, wf = quadratureRule(n)
            assert len(bp) == n
            assert len(wf) == n

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_exactness_polynomials(self, n):
        """n-point rule should exactly integrate polynomials up to degree 2n-1."""
        bp, wf = quadratureRule(n)
        for k in range(2 * n):
            # Integral of x^k over [-1, 1]
            numerical = np.sum(wf * bp**k)
            if k % 2 == 0:
                exact = 2.0 / (k + 1)
            else:
                exact = 0.0
            assert pytest.approx(numerical, abs=1e-8) == exact

    def test_n_equals_2(self):
        """Verify known 2-point rule: ±1/sqrt(3) with weight 1."""
        bp, wf = quadratureRule(2)
        bp_sorted = np.sort(bp)
        expected_bp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        np.testing.assert_allclose(bp_sorted, expected_bp, atol=1e-10)
        np.testing.assert_allclose(wf, np.array([1.0, 1.0]), atol=1e-10)


# ---------------------------------------------------------------------------
# Integration (zi_and_xi, rho_integral)
# ---------------------------------------------------------------------------


class TestZiAndXi:
    def test_output_shapes(self):
        """Verify output shapes for given nIP."""
        n1 = Normal("N1", 0, 1)
        n2 = Normal("N2", 0, 1)
        nIP = 8
        Z1, Z2, X1, X2, WIP, detJ = zi_and_xi(n1, n2, zmax=6, nIP=nIP)
        assert Z1.shape == (nIP, nIP)
        assert Z2.shape == (nIP, nIP)
        assert X1.shape == (nIP, nIP)
        assert X2.shape == (nIP, nIP)
        assert WIP.shape == (nIP, nIP)
        assert np.isscalar(detJ)

    def test_detJ_value(self):
        """Verify the Jacobian determinant is computed correctly."""
        n1 = Normal("N1", 0, 1)
        n2 = Normal("N2", 0, 1)
        zmax = 6
        _, _, _, _, _, detJ = zi_and_xi(n1, n2, zmax=zmax, nIP=4)
        expected = (zmax - (-zmax)) ** 2 / 4
        assert pytest.approx(detJ, abs=1e-10) == expected


class TestRhoIntegral:
    def test_uncorrelated_normals(self):
        """For two standard normals with rho0=0, integral ≈ 0."""
        n1 = Normal("N1", 0, 1)
        n2 = Normal("N2", 0, 1)
        Z1, Z2, X1, X2, WIP, detJ = zi_and_xi(n1, n2, zmax=6, nIP=32)
        result = rho_integral(0.0, n1, n2, Z1, Z2, X1, X2, WIP, detJ)
        assert pytest.approx(result, abs=1e-2) == 0.0

    def test_correlated_normals(self):
        """For two standard normals with rho0=0.5, integral ≈ 0.5."""
        n1 = Normal("N1", 0, 1)
        n2 = Normal("N2", 0, 1)
        Z1, Z2, X1, X2, WIP, detJ = zi_and_xi(n1, n2, zmax=6, nIP=32)
        result = rho_integral(0.5, n1, n2, Z1, Z2, X1, X2, WIP, detJ)
        assert pytest.approx(result, abs=1e-2) == 0.5

    def test_high_correlation(self):
        """For rho0=0.9, integral ≈ 0.9 for standard normals."""
        n1 = Normal("N1", 0, 1)
        n2 = Normal("N2", 0, 1)
        Z1, Z2, X1, X2, WIP, detJ = zi_and_xi(n1, n2, zmax=6, nIP=64)
        result = rho_integral(0.9, n1, n2, Z1, Z2, X1, X2, WIP, detJ)
        assert pytest.approx(result, abs=1e-2) == 0.9
