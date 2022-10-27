import pytest
import pystra as ra


def test_zero_inflated():
    """
    Test for the zero-inflated distribution
    """

    ZIN = ra.ZeroInflated("ZIN", ra.Normal("N", 5, 1), p=0.5)
    assert ZIN.mean == 2.5
    assert pytest.approx(ZIN.stdv, abs=1e-3) == 2.598
    assert pytest.approx(ZIN.pdf(0), abs=1e-6) == 0.5
