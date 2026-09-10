"""Meinen & Steenbergen (2025), Structural Safety 112, 102521, example 1.

DOI: 10.1016/j.strusafe.2024.102521. The fully dependent system consists
of identical events on one shared pair of exponential random variables.
Mixed-order alpha aggregation below intentionally reproduces the invalid
coordinate mixing discussed in the paper; it is not a SystemFORM operation.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import expon, norm

import pystra as ra


def _model(copula):
    return ra.StochasticModel(
        ra.JointDistribution(
            [
                ra.ScipyDist("X1", expon(scale=1)),
                ra.ScipyDist("X2", expon(scale=1 / 3)),
            ],
            copula,
        )
    )


def _system(copula, order, *, method="rosenblatt", distinct=False):
    options = ra.FORMOptions(transform=method, rosenblatt_order=order)
    first = lambda X1, X2: 8 * X1 + 2 * X2 - 1
    second = (lambda X1, X2: X2 - 0.5 * X1) if distinct else first
    analysis = ra.SystemFORM(
        _model(copula),
        ra.SeriesSystem([ra.Component("one", first), ra.Component("two", second)]),
        options=options,
    )
    analysis.run()
    return analysis


@pytest.mark.parametrize(
    "copula, published_probabilities, mixed_probability, reference",
    [
        (ra.GaussianCopula([[1, 0.5], [0.5, 1]]), [0.098, 0.098], 0.133, 0.087184649),
        (ra.FrankCopula(4.73), [0.102, 0.114], 0.156, 0.093225253),
    ],
)
def test_meinen_identical_events_preserve_identity_but_frank_depends_on_order(
    copula, published_probabilities, mixed_probability, reference
):
    canonical = _system(copula, [0, 1])
    reverse = _system(copula, [1, 0])
    for analysis in (canonical, reverse):
        assert analysis.results_valid
        assert analysis.correlation[0, 1] == 1
        assert analysis.Pf == pytest.approx(analysis.probabilities[0], abs=1e-12)
    actual = [canonical.Pf, reverse.Pf]
    np.testing.assert_allclose(actual, published_probabilities, atol=5e-4, rtol=0)
    if isinstance(copula, ra.GaussianCopula):
        assert actual[0] == pytest.approx(actual[1], abs=1e-10)
    else:
        assert abs(actual[1] - actual[0]) / np.mean(actual) > 0.1

    # Original-event integral, independent of FORM and its conditioning order.
    # Integrate along each physical coordinate as a cross-check. Section 3.3
    # appears to transpose the Gaussian/Frank dependent-system CMC values.
    integrals = []
    for i, j in ((0, 1), (1, 0)):
        rates, weights = (1, 3), (8, 2)

        def integrand(x):
            values = np.empty(2)
            values[i] = -np.expm1(-rates[i] * x)
            values[j] = -np.expm1(-rates[j] * (1 - weights[i] * x) / weights[j])
            conditional = copula.rosenblatt(values, order=[i, j])[j]
            return rates[i] * np.exp(-rates[i] * x) * conditional

        integrals.append(quad(integrand, 0, 1 / weights[i], epsabs=1e-11)[0])
    np.testing.assert_allclose(integrals, reference, atol=1e-9, rtol=0)

    # A deliberate anti-example: original variable labels do not make two
    # different conditional-normal coordinate systems interchangeable.
    rho = float(canonical.alphas[0] @ reverse.alphas[0])
    assert rho < 0.9
    b1, b2 = canonical.betas[0], reverse.betas[0]
    intersection = quad(
        lambda u: norm.pdf(u) * norm.cdf((-b2 - rho * u) / np.sqrt(1 - rho * rho)),
        -np.inf,
        -b1,
        epsabs=1e-11,
    )[0]
    invalid_mixed = sum(actual) - intersection
    assert invalid_mixed == pytest.approx(mixed_probability, abs=5e-4)
    assert invalid_mixed > max(actual)


def test_common_gaussian_coordinates_preserve_distinct_system_events():
    # The paper's discussion/Fig. 7 replaces the second LSF by X2 - 0.5 X1.
    # A global Gaussian rotation must preserve both tangent-event correlations
    # and the system probability, including when the components differ.
    copula = ra.GaussianCopula([[1, 0.5], [0.5, 1]])
    runs = [
        _system(copula, order, method=method, distinct=True)
        for method, order in (
            ("nataf", None),
            ("rosenblatt", [0, 1]),
            ("rosenblatt", [1, 0]),
        )
    ]
    for result in runs[1:]:
        np.testing.assert_allclose(result.betas, runs[0].betas, atol=2e-6, rtol=0)
        np.testing.assert_allclose(
            result.correlation, runs[0].correlation, atol=2e-6, rtol=0
        )
        assert result.Pf == pytest.approx(runs[0].Pf, abs=2e-6)
