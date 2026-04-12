"""Tests for Active Learning reliability analysis.

Benchmark problems are drawn from the structural reliability literature:

- **R-S (M=5)**: Sum of 5 standard normals, threshold at 0.
  Reference: UQlab test suite (uq_Reliability_test_AKMCS_RS.m).
  Analytical beta = M/sqrt(M) = sqrt(5) ~ 2.236, Pf ~ 0.0190.

- **Four-branch series system**: Waarts (2000) / Schueremans & Van Gemert
  (2005). Two standard normal inputs, four limit state branches.
  Reference Pf ~ 4.46e-3, beta ~ 2.61.

- **Simply supported beam**: 5 lognormal inputs (b, h, L, E, p).
  Reference: UQlab test suite.
  Reference Pf = 0.0172, beta ~ 2.115.

References
----------
Schueremans, L. & Van Gemert, D. (2005). Benefit of splines and neural
    networks in simulation based structural reliability analysis.
    *Structural Safety*, 27(3), 246–261.
Moustapha, M. et al. (2022). Active learning for structural reliability:
    Survey, general framework and benchmark. *Structural Safety*, 96, 102174.
"""

import numpy as np
import pytest
import pystra as ra


# ---------------------------------------------------------------------------
# Benchmark problem definitions
# ---------------------------------------------------------------------------


def rs_sum_model():
    """Sum-of-normals R-S problem (M=5 standard normals, threshold=0).

    g(X) = sum(X) with X_i ~ N(0,1), failure when g <= 0.
    Since g ~ N(0, 5), we have Pf = Phi(-sqrt(5)) ~ 0.0127.

    However, UQlab's test uses threshold=0 with CompOp '<=' and
    reports PFReference = 0.0190 and beta = M/sqrt(M) = sqrt(5).
    For pystra, where failure is g <= 0, this is equivalent to
    g(X) = X1 + X2 + ... + X5 and Pf = Phi(-5/sqrt(5)) = Phi(-sqrt(5)).

    Analytical: beta = sqrt(5) ~ 2.236, Pf ~ 0.01267.
    """
    model = ra.StochasticModel()
    for i in range(1, 6):
        model.addVariable(ra.Normal(f"X{i}", 1, 1))
    limit_state = ra.LimitState(lambda X1, X2, X3, X4, X5: X1 + X2 + X3 + X4 + X5)
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    return model, limit_state, options


def fourbranch_model():
    """Four-branch series system (Waarts 2000, Schueremans & Van Gemert 2005).

    g(X1, X2) = min(g1, g2, g3, g4) with:
        g1 = 3 + 0.1*(X1-X2)^2 - (X1+X2)/sqrt(2)
        g2 = 3 + 0.1*(X1-X2)^2 + (X1+X2)/sqrt(2)
        g3 = (X1-X2) + 6/sqrt(2)
        g4 = (X2-X1) + 6/sqrt(2)

    X1, X2 ~ N(0, 1) independent.
    Reference Pf ~ 4.46e-3 (beta ~ 2.61).
    """

    def lsf(X1, X2):
        k = 6.0
        g1 = 3 + 0.1 * (X1 - X2) ** 2 - (X1 + X2) / np.sqrt(2)
        g2 = 3 + 0.1 * (X1 - X2) ** 2 + (X1 + X2) / np.sqrt(2)
        g3 = (X1 - X2) + k / np.sqrt(2)
        g4 = (X2 - X1) + k / np.sqrt(2)
        return np.minimum(np.minimum(g1, g2), np.minimum(g3, g4))

    model = ra.StochasticModel()
    model.addVariable(ra.Normal("X1", 0, 1))
    model.addVariable(ra.Normal("X2", 0, 1))
    limit_state = ra.LimitState(lsf)
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    return model, limit_state, options


def simply_supported_beam_model():
    """Simply supported beam (UQlab benchmark).

    Deflection at midspan:
        V = (5 * p * L^4) / (32 * E * b * h^3)

    Failure when V >= 15 mm = 0.015 m, i.e. g = 0.015 - V <= 0.

    All inputs are lognormal:
        b: mean=0.15 m, std=0.0075 m (5% CV)
        h: mean=0.3 m,  std=0.015 m  (5% CV)
        L: mean=5 m,    std=0.05 m   (1% CV)
        E: mean=30000 MPa, std=4500 MPa (15% CV)
        p: mean=0.01 MN/m, std=0.002 MN/m (20% CV)

    Reference Pf = 0.0172, beta ~ 2.115.
    """

    def lsf(b, h, L, E, p):
        V = (5 * p * L**4) / (32 * E * b * h**3)
        return 0.015 - V

    model = ra.StochasticModel()
    model.addVariable(ra.Lognormal("b", 0.15, 0.0075))
    model.addVariable(ra.Lognormal("h", 0.3, 0.015))
    model.addVariable(ra.Lognormal("L", 5.0, 0.05))
    model.addVariable(ra.Lognormal("E", 30000, 4500))
    model.addVariable(ra.Lognormal("p", 0.01, 0.002))
    limit_state = ra.LimitState(lsf)
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    return model, limit_state, options


# ---------------------------------------------------------------------------
# Benchmark: R-S sum-of-normals (M=5)
# ---------------------------------------------------------------------------


class TestBenchmarkRSSum:
    """AK-MCS on sum-of-normals, cf. UQlab uq_Reliability_test_AKMCS_RS."""

    def test_kriging_rs_sum(self):
        """Kriging AL should recover Pf within 10% (UQlab TH=1e-1 convention).

        Reference: UQlab uq_Reliability_test_AKMCS_RS.m uses TH=1e-1 on Pf.
        Analytical Pf = Phi(-sqrt(5)) ~ 0.01267.
        """
        np.random.seed(42)
        model, limit_state, options = rs_sum_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            surrogate="kriging",
            n_candidates=10_000,
        )
        al.run()

        # UQlab convention: |Pf - Pf_ref| < 0.1 * Pf_ref
        from scipy.stats import norm as _norm

        pf_ref = _norm.cdf(-5.0 / np.sqrt(5.0))  # ~ 0.01267
        assert al.Pf > 0
        assert abs(al.Pf - pf_ref) < 0.1 * pf_ref
        assert al.n_evals < 100


# ---------------------------------------------------------------------------
# Benchmark: Four-branch series system (Waarts / Schueremans & Van Gemert)
# ---------------------------------------------------------------------------


class TestBenchmarkFourBranch:
    """AK-MCS on the four-branch series system."""

    def test_kriging_fourbranch(self):
        """Kriging AL on four-branch series system.

        Reference Pf ~ 4.46e-3 (Schueremans & Van Gemert 2005).
        Tolerance: 10% relative error on Pf (UQlab convention).
        Widened to 20% here due to MCS noise with 10k candidates
        (~45 expected failures).
        """
        np.random.seed(42)
        model, limit_state, options = fourbranch_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            surrogate="kriging",
            n_candidates=10_000,
            max_iterations=100,
        )
        al.run()

        pf_ref = 4.46e-3
        assert al.Pf > 0
        assert abs(al.Pf - pf_ref) < 0.2 * pf_ref
        assert al.n_evals < 100

    def test_eff_rs_sum(self):
        """EFF learning function on R-S sum problem.

        Same reference as test_kriging_rs_sum, 10% Pf tolerance.
        """
        np.random.seed(42)
        model, limit_state, options = rs_sum_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            surrogate="kriging",
            learning_function="EFF",
            n_candidates=10_000,
        )
        al.run()

        from scipy.stats import norm as _norm

        pf_ref = _norm.cdf(-5.0 / np.sqrt(5.0))
        assert al.Pf > 0
        assert abs(al.Pf - pf_ref) < 0.1 * pf_ref


# ---------------------------------------------------------------------------
# Benchmark: Simply supported beam (UQlab benchmark, 5 lognormals)
# ---------------------------------------------------------------------------


class TestBenchmarkBeam:
    """AK-MCS on the simply supported beam problem."""

    def test_kriging_beam(self):
        """Kriging AL on simply supported beam (UQlab benchmark).

        Reference Pf = 0.0172 (UQlab, Eps=10%).
        Tolerance: 10% relative error on Pf.
        """
        np.random.seed(42)
        model, limit_state, options = simply_supported_beam_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            surrogate="kriging",
            n_candidates=20_000,
            max_iterations=100,
        )
        al.run()

        pf_ref = 0.0172
        assert al.Pf > 0
        assert abs(al.Pf - pf_ref) < 0.1 * pf_ref
        assert al.n_evals < 100


# ---------------------------------------------------------------------------
# Active Learning with PCE
# ---------------------------------------------------------------------------


class TestActiveLearningPCE:
    """Tests for AL with PCE surrogate on benchmark problems."""

    def test_pce_rs_sum(self):
        """PCE AL on R-S sum problem, 10% Pf tolerance."""
        pytest.importorskip("chaospy")
        np.random.seed(42)
        model, limit_state, options = rs_sum_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            surrogate="pce",
            n_candidates=10_000,
        )
        al.run()

        from scipy.stats import norm as _norm

        pf_ref = _norm.cdf(-5.0 / np.sqrt(5.0))
        assert al.Pf > 0
        assert abs(al.Pf - pf_ref) < 0.1 * pf_ref


# ---------------------------------------------------------------------------
# Structural / API tests
# ---------------------------------------------------------------------------


class TestActiveLearningAPI:
    """Tests for the AL public API and error handling."""

    def test_history_populated(self):
        """History dict should be populated after run."""
        np.random.seed(42)
        model, limit_state, options = rs_sum_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            n_candidates=5000,
        )
        al.run()

        assert len(al.history["beta"]) > 0
        assert len(al.history["Pf"]) == len(al.history["beta"])
        assert len(al.history["lf_best"]) == len(al.history["beta"])

    def test_getBeta_getFailure(self):
        """getBeta() and getFailure() should return consistent values."""
        np.random.seed(42)
        model, limit_state, options = rs_sum_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            n_candidates=5000,
        )
        al.run()

        assert al.getBeta() == al.beta
        assert al.getFailure() == al.Pf

    def test_invalid_surrogate_raises(self):
        """Unknown surrogate name should raise ValueError."""
        model, limit_state, options = rs_sum_model()
        with pytest.raises(ValueError, match="Unknown surrogate"):
            ra.ActiveLearning(
                analysis_options=options,
                stochastic_model=model,
                limit_state=limit_state,
                surrogate="neural_net",
            )

    def test_invalid_learning_function_raises(self):
        """Unknown learning function should raise ValueError."""
        model, limit_state, options = rs_sum_model()
        with pytest.raises(ValueError, match="Unknown learning function"):
            ra.ActiveLearning(
                analysis_options=options,
                stochastic_model=model,
                limit_state=limit_state,
                learning_function="INVALID",
            )

    def test_surrogate_model_accessible(self):
        """After run, the fitted surrogate should be accessible."""
        np.random.seed(42)
        model, limit_state, options = rs_sum_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            n_candidates=5000,
        )
        al.run()

        assert al.surrogate_model is not None
        mean, std = al.surrogate_model.predict(np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]))
        assert mean.shape == (1,)
        assert std.shape == (1,)

    def test_show_results(self, capsys):
        """showResults() should print without error."""
        np.random.seed(42)
        model, limit_state, options = rs_sum_model()

        al = ra.ActiveLearning(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            n_candidates=5000,
        )
        al.run()
        al.showResults()

        captured = capsys.readouterr()
        assert "ACTIVE LEARNING" in captured.out
        assert "beta" in captured.out.lower()
