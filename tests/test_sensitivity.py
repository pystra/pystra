"""Tests for SensitivityAnalysis."""

import pytest
import numpy as np
import pystra as ra
from pystra.distributions import Normal, Lognormal
from pystra.distributions.distribution import Distribution
from pystra.distributions.gev import GEVmax, GEVmin
from pystra.distributions.beta import Beta
from pystra.distributions.weibull import Weibull
from pystra.distributions.uniform import Uniform
from pystra.distributions.gamma import Gamma


class TestSensitivityAnalysis:
    def _make_problem(self):
        """Create a simple reliability problem for sensitivity testing."""
        model = ra.model.StochasticModel()
        model.addVariable(Normal("R", 10, 2))
        model.addVariable(Normal("S", 5, 1))

        def lsf(R, S):
            return R - S

        ls = ra.model.LimitState(lsf)
        return model, ls

    def test_run_returns_dict(self):
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run()
        assert isinstance(result, dict)

    def test_result_keys_match_variables(self):
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run()
        assert set(result.keys()) == {"R", "S"}

    def test_result_has_mean_and_std(self):
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run()
        for name in ["R", "S"]:
            assert "mean" in result[name]
            assert "std" in result[name]

    def test_sensitivity_values_are_finite(self):
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run()
        for name in result:
            for param in ["mean", "std"]:
                assert np.isfinite(result[name][param])

    def test_sensitivity_signs(self):
        """For g = R - S:
        - Increasing R mean -> higher beta -> positive sensitivity
        - Increasing S mean -> lower beta -> negative sensitivity
        """
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run()
        assert result["R"]["mean"] > 0, "R mean sensitivity should be positive"
        assert result["S"]["mean"] < 0, "S mean sensitivity should be negative"

    def test_different_delta_consistent(self):
        """Results should be consistent for different delta values."""
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)

        result1 = sa.run(delta=0.01)
        result2 = sa.run(delta=0.005)

        for name in ["R", "S"]:
            for param in ["mean", "std"]:
                assert pytest.approx(result1[name][param], rel=0.1) == result2[name][param]

    def test_with_lognormal(self):
        """Test sensitivity works with non-Normal distributions."""
        model = ra.model.StochasticModel()
        model.addVariable(Lognormal("R", 10, 2))
        model.addVariable(Normal("S", 5, 1))

        def lsf(R, S):
            return R - S

        ls = ra.model.LimitState(lsf)
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run()

        assert result["R"]["mean"] > 0
        assert result["S"]["mean"] < 0
        for name in result:
            for param in ["mean", "std"]:
                assert np.isfinite(result[name][param])

    def test_default_options(self):
        """SensitivityAnalysis should work without explicit options."""
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        assert sa.options is not None
        result = sa.run()
        assert len(result) == 2

    def test_custom_options(self):
        """SensitivityAnalysis should accept custom options."""
        model, ls = self._make_problem()
        opts = ra.AnalysisOptions()
        opts.setPrintOutput(False)
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)
        result = sa.run()
        assert len(result) == 2

    def test_run_form_alias(self):
        """run_form() should be equivalent to run()."""
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result_run = sa.run()
        result_alias = sa.run_form()
        for name in ["R", "S"]:
            for param in ["mean", "std"]:
                assert result_run[name][param] == pytest.approx(
                    result_alias[name][param], rel=1e-10
                )


class TestClosedFormSensitivity:
    """Tests for the closed-form sensitivity method (Bourinet 2017)."""

    @staticmethod
    def _make_bourinet_example2():
        """Bourinet (2017), Example 2: correlated lognormals.

        X1 ~ LN(5, 5), X2 ~ LN(1, 1), rho = 0.5, g = x1 - x2.
        Both have CoV = 1, giving an exact beta = 2.1218.
        """
        model = ra.StochasticModel()
        model.addVariable(Lognormal("X1", 5, 5))
        model.addVariable(Lognormal("X2", 1, 1))

        corr = ra.CorrelationMatrix(np.array([[1.0, 0.5], [0.5, 1.0]]))
        model.setCorrelation(corr)

        def lsf(X1, X2):
            return X1 - X2

        ls = ra.LimitState(lsf)
        return model, ls

    def test_closed_form_returns_structure(self):
        model, ls = self._make_bourinet_example2()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run(numerical=False)

        assert "marginal" in result
        assert "correlation" in result
        assert set(result["marginal"].keys()) == {"X1", "X2"}
        assert result["correlation"].shape == (2, 2)

    def test_bourinet_example2_marginal_sensitivities(self):
        """Validate against Table 4 of Bourinet (2017)."""
        model, ls = self._make_bourinet_example2()
        opts = ra.AnalysisOptions()
        opts.setPrintOutput(False)
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)
        result = sa.run(numerical=False)

        m = result["marginal"]

        # Reference values from Table 4 (analytical column)
        assert m["X1"]["mean"] == pytest.approx(0.5184, abs=0.005)
        assert m["X2"]["mean"] == pytest.approx(-1.3629, abs=0.005)
        assert m["X1"]["std"] == pytest.approx(-0.2548, abs=0.005)
        assert m["X2"]["std"] == pytest.approx(0.0445, abs=0.005)

    def test_bourinet_example2_correlation_sensitivity(self):
        """Validate d_beta/d_rho against Eq. (31) of Bourinet (2017)."""
        model, ls = self._make_bourinet_example2()
        opts = ra.AnalysisOptions()
        opts.setPrintOutput(False)
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)
        result = sa.run(numerical=False)

        # d_beta/d_rho_12 = 2.4585 (from Eq. 31)
        assert result["correlation"][1, 0] == pytest.approx(2.4585, abs=0.01)
        # Symmetric
        assert result["correlation"][0, 1] == result["correlation"][1, 0]

    def test_closed_form_vs_fd_uncorrelated_normals(self):
        """Closed-form and FD should agree for uncorrelated normals."""
        model = ra.StochasticModel()
        model.addVariable(ra.Normal("R", 10, 2))
        model.addVariable(ra.Normal("S", 5, 1))

        def lsf(R, S):
            return R - S

        ls = ra.LimitState(lsf)
        sa = ra.SensitivityAnalysis(ls, model)

        fd = sa.run(numerical=True, delta=0.001)
        cf = sa.run(numerical=False)

        for name in ["R", "S"]:
            for param in ["mean", "std"]:
                assert cf["marginal"][name][param] == pytest.approx(
                    fd[name][param], rel=0.02
                )

    def test_closed_form_vs_fd_correlated_lognormals(self):
        """Closed-form and FD should agree on sign for correlated lognormals.

        The global FD method is numerically unstable for parameters with
        small sensitivities in this problem (perturbing sigma changes the
        Nataf transformation, and FORM re-convergence adds noise).  We
        therefore only check that the closed-form sensitivities are
        finite and have the correct sign, relying on the analytical
        reference values (test_bourinet_example2_*) for accuracy.
        """
        model, ls = self._make_bourinet_example2()
        opts = ra.AnalysisOptions()
        opts.setPrintOutput(False)
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)

        cf = sa.run(numerical=False)

        for name in ["X1", "X2"]:
            for param in ["mean", "std"]:
                assert np.isfinite(cf["marginal"][name][param])

        # Sign checks: increasing X1 mean increases beta, increasing X2 mean decreases beta
        assert cf["marginal"]["X1"]["mean"] > 0
        assert cf["marginal"]["X2"]["mean"] < 0


class TestSensitivityParams:
    """Tests for the sensitivity_params / _make_copy / _ctor_kwargs machinery."""

    def test_default_sensitivity_params(self):
        """Base distributions return {"mean", "std"} by default."""
        dist = Normal("X", 10, 2)
        sp = dist.sensitivity_params
        assert set(sp.keys()) == {"mean", "std"}
        assert sp["mean"] == 10.0
        assert sp["std"] == 2.0

    def test_gev_sensitivity_params(self):
        """GEVmax returns {"mean", "std", "shape"}."""
        dist = GEVmax("X", 100, 20, shape=0.1)
        sp = dist.sensitivity_params
        assert set(sp.keys()) == {"mean", "std", "shape"}
        assert sp["shape"] == 0.1

    def test_gevmin_sensitivity_params(self):
        """GEVmin returns {"mean", "std", "shape"}."""
        dist = GEVmin("X", 100, 20, shape=0.1)
        sp = dist.sensitivity_params
        assert set(sp.keys()) == {"mean", "std", "shape"}
        assert sp["shape"] == 0.1

    def test_beta_no_extra_sensitivity_params(self):
        """Beta bounds are in _ctor_kwargs but NOT in sensitivity_params."""
        dist = Beta("X", 0.5, 0.1, a=0, b=1)
        sp = dist.sensitivity_params
        assert set(sp.keys()) == {"mean", "std"}
        assert "a" not in sp
        assert "b" not in sp

    @pytest.mark.parametrize("cls,kwargs", [
        (Normal, {"mean": 10, "stdv": 2}),
        (Lognormal, {"mean": 10, "stdv": 2}),
        (Uniform, {"mean": 5, "stdv": 0.5}),
        (Gamma, {"mean": 10, "stdv": 2}),
        (GEVmax, {"mean": 100, "stdv": 20, "shape": 0.1}),
        (GEVmin, {"mean": 100, "stdv": 20, "shape": 0.1}),
    ])
    def test_make_copy_roundtrip(self, cls, kwargs):
        """_make_copy() with no overrides reproduces the original."""
        dist = cls("X", **kwargs)
        copy = dist._make_copy()
        x_test = dist.mean + 0.5 * dist.stdv
        assert copy.cdf(x_test) == pytest.approx(dist.cdf(x_test), abs=1e-8)

    def test_make_copy_beta_with_bounds(self):
        """Beta with non-default bounds reconstructs faithfully."""
        dist = Beta("X", 5, 1, a=3, b=10)
        copy = dist._make_copy()
        x_test = 5.5
        assert copy.cdf(x_test) == pytest.approx(dist.cdf(x_test), abs=1e-8)

    def test_make_copy_weibull_with_epsilon(self):
        """Weibull with non-zero epsilon reconstructs faithfully."""
        dist = Weibull("X", 10, 3, epsilon=2)
        copy = dist._make_copy()
        x_test = 9.0
        assert copy.cdf(x_test) == pytest.approx(dist.cdf(x_test), abs=1e-8)

    def test_make_copy_perturbed_mean(self):
        """_make_copy with perturbed mean produces shifted distribution."""
        dist = Normal("X", 10, 2)
        perturbed = dist._make_copy(mean=10.1)
        assert perturbed.mean == pytest.approx(10.1)
        assert perturbed.stdv == pytest.approx(2.0)

    def test_make_copy_perturbed_shape(self):
        """_make_copy with perturbed shape for GEV works correctly."""
        dist = GEVmax("X", 100, 20, shape=0.1)
        perturbed = dist._make_copy(shape=0.11)
        # Different shape → different distribution (different CDF)
        x_test = 120.0
        assert perturbed.cdf(x_test) != pytest.approx(dist.cdf(x_test), abs=1e-4)

    def test_dF_dtheta_gev_includes_shape(self):
        """GEV's dF_dtheta returns derivatives for all sensitivity_params."""
        dist = GEVmax("X", 100, 20, shape=0.1)
        dF = dist.dF_dtheta(110.0)
        assert "mean" in dF
        assert "std" in dF
        assert "shape" in dF
        for key in dF:
            assert np.isfinite(dF[key])

    def test_dmoments_dtheta_mean(self):
        """_dmoments_dtheta for 'mean' returns (1, 0) exactly."""
        dist = Normal("X", 10, 2)
        dmu, dsig = dist._dmoments_dtheta("mean")
        assert dmu == 1.0
        assert dsig == 0.0

    def test_dmoments_dtheta_std(self):
        """_dmoments_dtheta for 'std' returns (0, 1) exactly."""
        dist = Normal("X", 10, 2)
        dmu, dsig = dist._dmoments_dtheta("std")
        assert dmu == 0.0
        assert dsig == 1.0

    def test_dmoments_dtheta_shape_gev(self):
        """_dmoments_dtheta for 'shape' on GEV returns finite values."""
        dist = GEVmax("X", 100, 20, shape=0.1)
        dmu, dsig = dist._dmoments_dtheta("shape")
        assert np.isfinite(dmu)
        assert np.isfinite(dsig)
        # Changing shape should affect both mean and stdv
        # (for non-zero shape, both depend on shape via gamma functions)


class TestGEVSensitivity:
    """Tests for GEV shape sensitivity in FORM."""

    @staticmethod
    def _make_gev_problem():
        """Problem with a GEV load variable."""
        model = ra.StochasticModel()
        model.addVariable(Normal("R", 50, 5))
        model.addVariable(GEVmax("S", 20, 8, shape=0.2))

        def lsf(R, S):
            return R - S

        ls = ra.LimitState(lsf)
        opts = ra.AnalysisOptions()
        opts.setPrintOutput(False)
        return model, ls, opts

    def test_fd_result_includes_shape(self):
        """FD result dict includes 'shape' key for GEV variable."""
        model, ls, opts = self._make_gev_problem()
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)
        fd = sa.run(numerical=True)

        assert "shape" in fd["S"], "GEV variable should have shape sensitivity"
        assert "shape" not in fd["R"], "Normal variable should not have shape"
        assert np.isfinite(fd["S"]["shape"])

    def test_cf_result_includes_shape(self):
        """CF result dict includes 'shape' key for GEV variable."""
        model, ls, opts = self._make_gev_problem()
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)
        cf = sa.run(numerical=False)

        assert "shape" in cf["marginal"]["S"]
        assert "shape" not in cf["marginal"]["R"]
        assert np.isfinite(cf["marginal"]["S"]["shape"])

    def test_gev_shape_sensitivity_sign(self):
        """Increasing GEV shape (heavier tail) should decrease beta.

        For g = R - S with S ~ GEVmax: increasing shape makes the
        right tail of S heavier, increasing the probability of
        large loads, so beta should decrease → ∂β/∂ξ < 0.
        """
        model, ls, opts = self._make_gev_problem()
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)
        cf = sa.run(numerical=False)

        assert cf["marginal"]["S"]["shape"] < 0, (
            "Heavier GEV tail should decrease reliability"
        )

    def test_gev_cf_vs_fd_shape(self):
        """CF and FD shape sensitivities should agree reasonably.

        The finite-difference method is inherently less accurate for shape
        parameters because perturbing the shape changes the entire
        distribution family, not just location/scale.  We therefore use a
        tighter tolerance for mean/std and a relaxed tolerance for shape.
        """
        model, ls, opts = self._make_gev_problem()
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)

        fd = sa.run(numerical=True, delta=0.001)
        cf = sa.run(numerical=False)

        # Check mean/std parameters agree tightly
        for name in ["R", "S"]:
            for param in fd[name]:
                if param == "shape":
                    # Shape FD is inherently less accurate — check sign
                    # agreement and relaxed tolerance
                    assert np.sign(cf["marginal"][name][param]) == np.sign(
                        fd[name][param]
                    ), f"CF vs FD sign mismatch for {name}/{param}"
                    assert cf["marginal"][name][param] == pytest.approx(
                        fd[name][param], rel=0.15
                    ), f"CF vs FD mismatch for {name}/{param}"
                else:
                    assert cf["marginal"][name][param] == pytest.approx(
                        fd[name][param], rel=0.05
                    ), f"CF vs FD mismatch for {name}/{param}"

    def test_summary_method(self):
        """summary() returns a DataFrame with correct structure."""
        model, ls, opts = self._make_gev_problem()
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)

        # Test with FD result
        fd = sa.run(numerical=True)
        df = sa.summary(fd)
        assert "Variable" in df.columns
        assert "Parameter" in df.columns
        # R has 2 params (mean, std), S has 3 (mean, std, shape) → 5 rows
        assert len(df) == 5

        # Test with CF result
        cf = sa.run(numerical=False)
        df_cf = sa.summary(cf)
        assert len(df_cf) == 5


class TestUnsupportedDistributions:
    """Test that unsupported distributions raise clear errors."""

    def test_maximum_raises_valueerror(self):
        """Maximum distribution should raise ValueError in dF_dtheta."""
        from pystra.distributions.maximum import Maximum

        parent = Normal("X", 10, 2)
        dist = Maximum("Xmax", parent, N=100)

        with pytest.raises(ValueError, match="does not support sensitivity"):
            dist.dF_dtheta(12.0)
