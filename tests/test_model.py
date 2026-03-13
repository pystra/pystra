"""Tests for StochasticModel, LimitState, and CorrelationMatrix."""

import pytest
import numpy as np
import pystra as ra
from pystra.distributions import Normal, Lognormal, Constant
from pystra.correlation import CorrelationMatrix, computeModifiedCorrelationMatrix


# ---------------------------------------------------------------------------
# StochasticModel
# ---------------------------------------------------------------------------

class TestStochasticModel:
    def test_add_distribution_variable(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        assert len(model.getVariables()) == 1
        assert "X1" in model.getNames()

    def test_add_multiple_variables(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Normal("X2", 5, 1))
        model.addVariable(Normal("X3", 3, 0.5))
        assert len(model.getVariables()) == 3
        assert model.getNames() == ["X1", "X2", "X3"]

    def test_add_constant(self):
        model = ra.model.StochasticModel()
        model.addVariable(Constant("c", 5.0))
        assert model.getConstants() == {"c": 5.0}
        assert len(model.getVariables()) == 0

    def test_add_mixed(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Constant("c", 3.0))
        model.addVariable(Normal("X2", 5, 1))
        assert len(model.getVariables()) == 2
        assert model.getConstants() == {"c": 3.0}
        assert model.getNames() == ["X1", "c", "X2"]

    def test_duplicate_name_raises(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        with pytest.raises(Exception, match="already exists"):
            model.addVariable(Normal("X1", 5, 1))

    def test_invalid_type_raises(self):
        model = ra.model.StochasticModel()
        with pytest.raises(Exception):
            model.addVariable("not_a_distribution")

    def test_get_variable(self):
        model = ra.model.StochasticModel()
        n = Normal("X1", 10, 2)
        model.addVariable(n)
        assert model.getVariable("X1") is n

    def test_get_marginal_distributions(self):
        model = ra.model.StochasticModel()
        n1 = Normal("X1", 10, 2)
        n2 = Normal("X2", 5, 1)
        model.addVariable(n1)
        model.addVariable(n2)
        marg = model.getMarginalDistributions()
        assert len(marg) == 2
        assert marg[0] is n1
        assert marg[1] is n2

    def test_default_correlation_is_identity(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Normal("X2", 5, 1))
        corr = model.getCorrelation()
        np.testing.assert_array_equal(corr, np.eye(2))

    def test_set_correlation(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Normal("X2", 5, 1))
        C = CorrelationMatrix([[1.0, 0.5], [0.5, 1.0]])
        model.setCorrelation(C)
        expected = np.array([[1.0, 0.5], [0.5, 1.0]])
        np.testing.assert_array_equal(model.getCorrelation(), expected)

    def test_call_function_counter(self):
        model = ra.model.StochasticModel()
        assert model.getCallFunction() == 0
        model.addCallFunction(5)
        assert model.getCallFunction() == 5
        model.addCallFunction(3)
        assert model.getCallFunction() == 8

    # ---- Property access tests ----

    def test_properties_match_getters(self):
        """Properties return the same objects as the legacy getter methods."""
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Constant("c", 3.0))
        model.addVariable(Normal("X2", 5, 1))

        assert model.constants is model.getConstants()
        assert model.names is model.getNames()
        assert model.marginal_distributions is model.getMarginalDistributions()
        assert model.n_marg == model.getLenMarginalDistributions()
        assert model.correlation is model.getCorrelation()
        assert model.call_function == model.getCallFunction()

    def test_correlation_property_setter(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Normal("X2", 5, 1))
        new_corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        model.correlation = new_corr
        np.testing.assert_array_equal(model.correlation, new_corr)
        np.testing.assert_array_equal(model.getCorrelation(), new_corr)

    def test_modified_correlation_property(self):
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Normal("X2", 5, 1))
        Ro = np.eye(2) * 0.9
        model.modified_correlation = Ro
        assert model.modified_correlation is model.getModifiedCorrelation()

    def test_call_function_property_setter(self):
        model = ra.model.StochasticModel()
        model.call_function = 42
        assert model.call_function == 42
        assert model.getCallFunction() == 42


# ---------------------------------------------------------------------------
# LimitState
# ---------------------------------------------------------------------------

class TestLimitState:
    def test_construction_with_lambda(self):
        ls = ra.model.LimitState(lambda X1, X2: X1 - X2)
        assert ls.getExpression() is not None

    def test_construction_with_function(self):
        def lsf(R, S):
            return R - S
        ls = ra.model.LimitState(lsf)
        assert ls.getExpression() is lsf

    def test_set_expression(self):
        ls = ra.model.LimitState(lambda X1: X1)
        new_expr = lambda X1, X2: X1 - X2
        ls.setExpression(new_expr)
        assert ls.getExpression() is new_expr

    def test_evaluate_lsf_no_gradient(self):
        def lsf(R, S):
            return R - S

        model = ra.model.StochasticModel()
        model.addVariable(Normal("R", 10, 2))
        model.addVariable(Normal("S", 5, 1))

        ls = ra.model.LimitState(lsf)
        options = ra.AnalysisOptions()
        options.setPrintOutput(False)

        # evaluate_nogradient only processes nx > 1, so provide multiple samples
        x = np.array([[10.0, 12.0], [5.0, 3.0]])
        G, grad_G = ls.evaluate_lsf(x, model, options, diff_mode="no")
        # G should be R - S = [10-5, 12-3] = [5, 9]
        assert pytest.approx(G[0, 0], abs=1e-10) == 5.0
        assert pytest.approx(G[0, 1], abs=1e-10) == 9.0

    def test_evaluate_lsf_with_constants(self):
        def lsf(R, S, c):
            return c * R - S

        model = ra.model.StochasticModel()
        model.addVariable(Normal("R", 10, 2))
        model.addVariable(Normal("S", 5, 1))
        model.addVariable(Constant("c", 2.0))

        ls = ra.model.LimitState(lsf)
        options = ra.AnalysisOptions()
        options.setPrintOutput(False)

        # evaluate_nogradient only processes nx > 1
        x = np.array([[10.0, 8.0], [5.0, 3.0]])
        G, grad_G = ls.evaluate_lsf(x, model, options, diff_mode="no")
        # G should be c*R - S = [2*10-5, 2*8-3] = [15, 13]
        assert pytest.approx(G[0, 0], abs=1e-10) == 15.0
        assert pytest.approx(G[0, 1], abs=1e-10) == 13.0

    def test_evaluate_lsf_ffd_gradient(self):
        def lsf(R, S):
            return R - S

        model = ra.model.StochasticModel()
        model.addVariable(Normal("R", 10, 2))
        model.addVariable(Normal("S", 5, 1))

        ls = ra.model.LimitState(lsf)
        options = ra.AnalysisOptions()
        options.setPrintOutput(False)

        x = np.array([[10.0], [5.0]])
        G, grad_G = ls.evaluate_lsf(x, model, options)
        # Gradient should be approximately [1, -1] (dG/dR=1, dG/dS=-1)
        assert pytest.approx(grad_G[0, 0], abs=1e-2) == 1.0
        assert pytest.approx(grad_G[1, 0], abs=1e-2) == -1.0


# ---------------------------------------------------------------------------
# CorrelationMatrix
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    def test_construction(self):
        C = CorrelationMatrix([[1.0, 0.5], [0.5, 1.0]])
        assert C.getMatrix() == [[1.0, 0.5], [0.5, 1.0]]

    def test_getitem(self):
        C = CorrelationMatrix([[1.0, 0.3], [0.3, 1.0]])
        assert C[0][0] == 1.0
        assert C[0][1] == 0.3

    def test_repr(self):
        C = CorrelationMatrix([[1.0, 0.0], [0.0, 1.0]])
        r = repr(C)
        assert "1.0" in r

    def test_modified_correlation_uncorrelated_normals(self):
        """For uncorrelated normals, modified correlation should be identity."""
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Normal("X2", 5, 1))
        # Default correlation is identity
        Ro = computeModifiedCorrelationMatrix(model)
        np.testing.assert_allclose(Ro, np.eye(2), atol=1e-4)

    def test_modified_correlation_correlated_normals(self):
        """For correlated normals, modified ≈ original (Nataf preserves Normal correlations)."""
        model = ra.model.StochasticModel()
        model.addVariable(Normal("X1", 10, 2))
        model.addVariable(Normal("X2", 5, 1))
        C = CorrelationMatrix([[1.0, 0.5], [0.5, 1.0]])
        model.setCorrelation(C)
        Ro = computeModifiedCorrelationMatrix(model)
        np.testing.assert_allclose(Ro, np.array([[1.0, 0.5], [0.5, 1.0]]), atol=1e-2)
