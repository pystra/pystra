"""Geometry, analytic events and reference checks for Strong Maximum Test."""

import numpy as np
import pytest
import pystra as ra


def model(dimension=2):
    result = ra.StochasticModel()
    for i in range(dimension):
        result.add_variable(ra.Normal(f"X{i}", 0, 1))
    return result


def raw(function=lambda X0, **kwargs: 3 - X0, dimension=2, beta=3, **kwargs):
    return ra.StrongMaximumTest(
        stochastic_model=model(dimension),
        limit_state=ra.LimitState(function),
        design_point=[beta] + [0] * (dimension - 1),
        **kwargs,
    )


def test_openturns_reference_geometry_and_ceiling():
    # OpenTURNS t_StrongMaximumTest_std: beta=sqrt(10)-0.3,
    # epsilon=.01, tau=2; reported radius 5.481097, cosine .522209.
    result = raw(
        beta=np.sqrt(10) - 0.3,
        importance_level=0.01,
        accuracy_level=2,
        confidence_level=0.999999,
    )
    assert result.delta_epsilon == pytest.approx(0.45747116705246027)
    assert result.radius == pytest.approx(5.481097, abs=5e-7)
    assert result.vicinity_cosine == pytest.approx(0.522209, abs=5e-7)
    cap = np.arccos((1 + result.delta_epsilon) / (1 + 2 * result.delta_epsilon)) / np.pi
    assert result.cap_probability == pytest.approx(cap)
    # OpenTURNS rounds to 54; 55 is needed to meet the requested confidence.
    assert result.point_number == 55
    assert result.confidence_level >= 0.999999
    assert -np.expm1(54 * np.log1p(-cap)) < 0.999999


def test_five_dimensional_reference_geometry():
    result = raw(
        dimension=5, importance_level=0.01, accuracy_level=2, confidence_level=0.99
    )
    assert result.delta_epsilon == pytest.approx(0.42245251324668054)
    assert result.point_number == 125  # table rounds to 124
    fixed = raw(dimension=5, importance_level=0.01, accuracy_level=2, point_number=125)
    assert fixed.confidence_level == result.confidence_level


def test_three_dimensional_cap_area():
    result = raw(dimension=3)
    cosine = (1 + result.delta_epsilon) / (
        1 + result.accuracy_level * result.delta_epsilon
    )
    assert result.cap_probability == pytest.approx((1 - cosine) / 2)


def test_linear_plane_has_no_competing_region_and_partitions_sample():
    result = raw(point_number=1000, seed=17)
    result.run()
    assert result.status == "no_competing_region_detected"
    assert not result.has_competing_points
    assert result.get_points().shape == (0, 2)
    assert result.get_values().shape == (0,)
    assert result.masks["near_failure"].sum() > 0
    assert result.masks["far_safe"].sum() > 0
    assert not result.masks["near_safe"].any()
    np.testing.assert_array_equal(sum(result.masks.values()), np.ones(1000))
    np.testing.assert_allclose(np.linalg.norm(result.u_points, axis=1), result.radius)
    np.testing.assert_allclose(result.x_points, result.u_points, atol=1e-12)
    assert result.evaluation_count == 1002


def test_two_equal_modes_are_detected():
    result = raw(lambda X0, **kwargs: 9 - X0**2, point_number=500, seed=3)
    result.run()
    assert result.has_competing_points
    assert np.all(result.get_points()[:, 0] < -3)
    assert np.all(result.get_values() < 0)


def test_openturns_two_branch_event_detects_other_region():
    result = ra.StrongMaximumTest(
        stochastic_model=model(),
        limit_state=ra.LimitState(lambda X0, X1: 10 - (X0 - 0.3) ** 2 + X1**2),
        design_point=[0.3 - np.sqrt(10), 0],
        importance_level=0.01,
        accuracy_level=2,
        confidence_level=0.999999,
        seed=5,
    )
    result.run()
    assert result.has_competing_points
    assert np.all(result.get_points()[:, 0] > 0)


def test_post_form_reuses_correlated_transform_without_mutating_evaluator():
    m = model()
    m.set_correlation([[1, 0.6], [0.6, 1]])
    opts = ra.AnalysisOptions()
    opts.set_transform("svd")
    form = ra.FORM(
        stochastic_model=m,
        limit_state=ra.LimitState(lambda X0, X1: 3 - X1),
        analysis_options=opts,
    )
    form.run()
    last_x = form.limitstate.x.copy()
    design = form.get_design_point().copy()
    result = ra.StrongMaximumTest(form, point_number=200, seed=2)
    result.run()
    assert not result.has_competing_points
    np.testing.assert_array_equal(form.limitstate.x, last_x)
    np.testing.assert_array_equal(form.get_design_point(), design)
    recovered = np.array(
        [
            form.transform.x_to_u(x, m.get_marginal_distributions())
            for x in result.x_points
        ]
    )
    np.testing.assert_allclose(recovered, result.u_points, atol=1e-12)


def test_reproducible_local_rng_and_block_size():
    state = np.random.get_state()
    a = raw(point_number=120, seed=9)
    b = raw(point_number=120, seed=9)
    b.options.set_block_size(7)
    a.run()
    b.run()
    np.testing.assert_allclose(a.u_points, b.u_points)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_one_dimension():
    result = raw(dimension=1, point_number=20, seed=2)
    result.run()
    assert result.cap_probability == 0.5
    np.testing.assert_allclose(np.abs(result.u_points[:, 0]), result.radius)
    assert not result.has_competing_points


def test_bounded_island_illustrates_no_global_certificate():
    result = raw(
        lambda X0, X1: (X0 - 3) ** 2 + X1**2 - 0.25, beta=2.5, point_number=100, seed=4
    )
    result.run()
    assert result.radius > 3.5
    assert result.status == "no_competing_region_detected"
    assert not result.masks["near_failure"].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"importance_level": 0},
        {"importance_level": 1},
        {"importance_level": float("nan")},
        {"accuracy_level": 1},
        {"accuracy_level": float("inf")},
        {"confidence_level": 0},
        {"confidence_level": 1},
        {"confidence_level": float("nan")},
        {"point_number": 0},
        {"point_number": 2.5},
        {"point_number": True},
        {"point_number": 5, "confidence_level": 0.9},
        {"max_points": 0},
        {"beta": 0},
        {"beta": float("nan")},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        raw(**kwargs)


def test_budget_is_checked_before_model_evaluations():
    def no_call(**kwargs):
        raise AssertionError("must not evaluate")

    with pytest.raises(ValueError, match="exceeding max_points"):
        raw(no_call, dimension=20, max_points=100)


def test_bad_candidate_and_failed_rerun():
    result = raw(point_number=10, seed=2)
    result.run()
    result.limitstate.expression = lambda **kwargs: np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        result.run()
    assert not result.results_valid
    assert result.status == "failed"
    assert result.has_competing_points is None
    with pytest.raises(ValueError, match="no valid result"):
        result.get_points()
    with pytest.raises(ValueError, match="boundary"):
        raw(beta=2).run()
    with pytest.raises(ValueError, match="strictly safe"):
        raw(lambda X0, **kwargs: X0 - 3).run()


def test_unrun_form_rejected():
    with pytest.raises(ValueError, match="converged"):
        ra.StrongMaximumTest(ra.FORM())


def test_non_normal_post_form_evaluates_original_physical_function():
    m = ra.StochasticModel()
    m.add_variable(ra.Lognormal("R", np.exp(0.5), np.sqrt((np.exp(1) - 1) * np.exp(1))))
    m.add_variable(ra.Normal("S", 0, 1))
    m.add_variable(ra.Constant("C", 3))
    form = ra.FORM(
        stochastic_model=m, limit_state=ra.LimitState(lambda R, S, C: C - np.log(R))
    )
    form.run()
    check = ra.StrongMaximumTest(form, point_number=100, seed=12)
    check.run()
    np.testing.assert_allclose(check.values, 3 - np.log(check.x_points[:, 0]))
    np.testing.assert_allclose(
        check.x_points[:, 0], np.exp(check.u_points[:, 0]), rtol=1e-9
    )
    assert not check.has_competing_points


def test_positive_scaling_preserves_diagnostic():
    a = raw(lambda X0, **kw: 9 - X0**2, point_number=100, seed=7)
    b = raw(lambda X0, **kw: 1e-5 * (9 - X0**2), point_number=100, seed=7)
    a.run()
    b.run()
    for name in a.masks:
        np.testing.assert_array_equal(a.masks[name], b.masks[name])


def test_nonconverged_form_rejected():
    options = ra.AnalysisOptions()
    options.set_imax(1)
    form = ra.FORM(
        stochastic_model=model(),
        limit_state=ra.LimitState(lambda X0, X1: 3 - X0),
        analysis_options=options,
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        form.run()
    with pytest.raises(ValueError, match="converged"):
        ra.StrongMaximumTest(form)
