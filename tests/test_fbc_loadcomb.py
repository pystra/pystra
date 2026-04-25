# -*- coding: utf-8 -*-

import pytest

import pystra as ra


def test_fbc_process_maximum_uses_interval_count():
    parent = ra.Normal("Q", 10, 2)
    process = ra.FBCProcess("Q", parent=parent, basic_interval=0.25)

    maximum = process.maximum(duration=1.0)

    assert maximum.N == 4.0
    assert pytest.approx(maximum.cdf(12.0), abs=1e-8) == parent.cdf(12.0) ** 4


def test_fbc_process_from_maximum_recovers_parent_relation():
    annual_max = ra.Gumbel("Q", 10, 2)
    process = ra.FBCProcess.from_maximum(
        "Q", maximum=annual_max, maximum_duration=1.0, basic_interval=0.25
    )

    parent = process.point_in_time()

    assert parent.N == 4.0
    assert pytest.approx(parent.cdf(12.0), abs=1e-8) == annual_max.cdf(12.0) ** 0.25


def test_loadcombination_explicit_cases_builds_stochastic_model():
    R = ra.Normal("R", 100, 10)
    G = ra.Normal("G", 30, 3)
    Q = ra.Gumbel("Q", 20, 4)
    z = ra.Constant("z", 1.0)

    lc = ra.LoadCombination(
        lsf=lambda z, R, G, Q: z * R - G - Q,
        cases={"Q_leading": {"R": R, "G": G, "Q": Q}},
        constants={"z": z},
    )

    case = lc.case("Q_leading")
    sm = lc.stochastic_model("Q_leading")

    assert list(case.keys()) == ["R", "G", "Q"]
    assert sm.getNames() == ["z", "R", "G", "Q"]
    assert sm.getConstants()["z"] == 1.0
    assert lc.get_label("comb_cases") == ["Q_leading"]
    assert lc.get_num_comb() == 1


def test_loadcombination_turkstra_generates_leading_cases_from_fbc_processes():
    R = ra.Normal("R", 100, 10)
    G = ra.Normal("G", 30, 3)
    Q1 = ra.FBCProcess("Q1", ra.Normal("Q1", 10, 2), basic_interval=0.25)
    Q2 = ra.FBCProcess("Q2", ra.Normal("Q2", 8, 1), basic_interval=0.10)

    lc = ra.LoadCombination.turkstra(
        lsf=lambda R, G, Q1, Q2: R - G - Q1 - Q2,
        resistance={"R": R},
        permanent={"G": G},
        variable={"Q1": Q1, "Q2": Q2},
        reference_period=1.0,
    )

    q1_case = lc.case("Q1_leading")
    q2_case = lc.case("Q2_leading")

    assert q1_case["Q1"].N == 4.0
    assert q1_case["Q2"].N == 2.5
    assert q2_case["Q1"].N == 1.0
    assert q2_case["Q2"].N == 10.0
    assert lc.get_label("resist") == ["R"]
    assert lc.get_label("other") == ["G"]
    assert lc.get_label("comb_vrs") == ["Q1", "Q2"]
    assert lc.dict_comb_cases == {
        "Q1_leading": ["Q1"],
        "Q2_leading": ["Q2"],
    }


def test_loadcombination_explicit_cases_reject_fbc_processes():
    process = ra.FBCProcess("Q", ra.Normal("Q", 10, 2), basic_interval=0.25)

    with pytest.raises(Exception, match="Distribution or Constant"):
        ra.LoadCombination(cases={"invalid": {"Q": process}})


def test_turkstra_cases_match_sorensen_example_4_distribution_exponents():
    """Sørensen Note 10, Example 4: imposed load and wind load.

    The example has T = 1 year, tau_1 = 0.5 years, and tau_2 = 1 day,
    so r_1 = 2 and r_2 = 360.  Sørensen's JCSS short-course slides
    write the companion distributions in terms of annual-maximum
    distributions with a 1/r_1 exponent.  With annual maximum
    distributions supplied as input, Turkstra's rule with FBC processes
    gives:

    - Q1 leading: Q1 annual maximum, Q2 maximum over tau_1
      => F_Q2C = F_Q2,max ** (180 / 360)
    - Q2 leading: Q2 annual maximum, Q1 parent over tau_1
      => F_Q1C = F_Q1,max ** (1 / 2)
    """
    q = 1.2
    Q1max = ra.Gumbel("Q1", 1.0, 0.2)
    Q2max = ra.Gumbel("Q2", 1.0, 0.4)
    Q1 = ra.FBCProcess.from_maximum(
        "Q1", maximum=Q1max, maximum_duration=1.0, basic_interval=0.5
    )
    Q2 = ra.FBCProcess.from_maximum(
        "Q2", maximum=Q2max, maximum_duration=1.0, basic_interval=1 / 360
    )

    lc = ra.LoadCombination.turkstra(
        variable={"Q1": Q1, "Q2": Q2},
        reference_period=1.0,
    )
    q1_leading = lc.case("Q1_leading")
    q2_leading = lc.case("Q2_leading")

    assert pytest.approx(q1_leading["Q1"].cdf(q), abs=1e-8) == Q1max.cdf(q)
    assert (
        pytest.approx(q1_leading["Q2"].cdf(q), abs=1e-8)
        == Q2max.cdf(q) ** 0.5
    )
    assert (
        pytest.approx(q2_leading["Q1"].cdf(q), abs=1e-8)
        == Q1max.cdf(q) ** 0.5
    )
    assert pytest.approx(q2_leading["Q2"].cdf(q), abs=1e-8) == Q2max.cdf(q)


def test_loadcombination_legacy_inputs_are_normalized():
    R = ra.Normal("R", 100, 10)
    G = ra.Normal("G", 30, 3)
    Qmax = ra.Gumbel("Q", 20, 4)
    Qpit = ra.Normal("Q", 10, 2)

    with pytest.deprecated_call():
        lc = ra.LoadCombination(
            lsf=lambda R, G, Q: R - G - Q,
            dict_dist_comb={"Q": {"max": Qmax, "pit": Qpit}},
            list_dist_resist=[R],
            list_dist_other=[G],
            dict_comb_cases={"Q_max": ["Q"]},
        )

    case = lc.case("Q_max")

    assert list(case.keys()) == ["R", "G", "Q"]
    assert case["Q"] is Qmax
    assert lc.get_dict_dist_comb()["Q_max"]["Q"] is Qmax
