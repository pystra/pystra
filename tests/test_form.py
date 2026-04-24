import warnings

import numpy as np

from pystra.form import Form


def test_compute_gamma_uses_diagonal_without_offdiagonal_warning():
    form = Form()
    form.J = np.array([[1.0, 0.0], [-0.5, 1.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        form.computeGamma()

    expected = np.diag(np.sqrt(np.diag(form.J @ form.J.T)))
    np.testing.assert_allclose(form.gamma, expected)
