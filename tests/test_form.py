import warnings

import numpy as np

from pystra import LimitState, StochasticModel
from pystra.reliability.form import FORM


def test_compute_gamma_uses_diagonal_without_offdiagonal_warning():
    form = FORM(StochasticModel(), LimitState(lambda X: X))
    form._J = np.array([[1.0, 0.0], [-0.5, 1.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        form._compute_gamma()

    expected = np.diag(np.sqrt(np.diag(form._J @ form._J.T)))
    np.testing.assert_allclose(form._gamma, expected)
