import pytest
import pystra as ra
from scipy.stats import genextreme as gev


def test_scipy():
    """
    Test the scipy wrapper
    """

    def lsf(X1, X2, C):
        """
        Basic R-S
        """
        return X1 - X2 - C

    X2 = ra.ScipyDist("X2", gev(c=0.1, loc=200, scale=50))

    limit_state = ra.LimitState(lsf)

    model = ra.StochasticModel()
    model.addVariable(ra.Normal("X1", 500, 100))
    model.addVariable(X2)
    model.addVariable(ra.Constant("C", 50))

    form = ra.Form(stochastic_model=model, limit_state=limit_state)
    form.run()

    # validate results
    assert form.i == 5
    assert pytest.approx(form.beta[0], abs=1e-4) == 1.9610


def test_negative():
    """
    Test the negative wrapper distribution
    """
    mR, sR = 500, 100
    mS, sS = 200, 50

    def run_positive():
        def lsf(R, S):
            return R - S

        model = ra.StochasticModel()
        model.addVariable(ra.Normal("R", mR, sR))
        model.addVariable(ra.Normal("S", mS, sS))

        limit_state = ra.LimitState(lsf)

        form = ra.Form(stochastic_model=model, limit_state=limit_state)
        form.run()

        return form.i, form.beta

    def run_negative():
        def lsf(R, S):
            return R - abs(S)

        Spos = ra.Normal("Spos", mS, sS)
        S = ra.Negative("S", Spos)

        model = ra.StochasticModel()
        model.addVariable(ra.Normal("R", mR, sR))
        model.addVariable(S)

        limit_state = ra.LimitState(lsf)

        form = ra.Form(stochastic_model=model, limit_state=limit_state)
        form.run()

        return form.i, form.beta

    pos_i, pos_beta = run_positive()
    neg_i, neg_beta = run_negative()

    # validate results
    assert pos_i == neg_i
    assert pytest.approx(pos_beta, abs=1e-4) == neg_beta
