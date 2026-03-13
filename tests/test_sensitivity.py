"""Tests for SensitivityAnalysis."""

import pytest
import numpy as np
import pystra as ra
from pystra.distributions import Normal, Lognormal


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
        result = sa.run_form()
        assert isinstance(result, dict)

    def test_result_keys_match_variables(self):
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run_form()
        assert set(result.keys()) == {"R", "S"}

    def test_result_has_mean_and_std(self):
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run_form()
        for name in ["R", "S"]:
            assert "mean" in result[name]
            assert "std" in result[name]

    def test_sensitivity_values_are_finite(self):
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run_form()
        for name in result:
            for param in ["mean", "std"]:
                assert np.isfinite(result[name][param])

    def test_sensitivity_signs(self):
        """For g = R - S:
        - Increasing R mean → higher beta → positive sensitivity
        - Increasing S mean → lower beta → negative sensitivity
        """
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)
        result = sa.run_form()
        assert result["R"]["mean"] > 0, "R mean sensitivity should be positive"
        assert result["S"]["mean"] < 0, "S mean sensitivity should be negative"

    def test_different_delta_consistent(self):
        """Results should be consistent for different delta values."""
        model, ls = self._make_problem()
        sa = ra.SensitivityAnalysis(ls, model)

        result1 = sa.run_form(delta=0.01)
        result2 = sa.run_form(delta=0.005)

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
        result = sa.run_form()

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
        result = sa.run_form()
        assert len(result) == 2

    def test_custom_options(self):
        """SensitivityAnalysis should accept custom options."""
        model, ls = self._make_problem()
        opts = ra.AnalysisOptions()
        opts.setPrintOutput(False)
        sa = ra.SensitivityAnalysis(ls, model, analysis_options=opts)
        result = sa.run_form()
        assert len(result) == 2
