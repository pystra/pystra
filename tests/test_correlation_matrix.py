"""A validated CorrelationMatrix, with Cholesky, nearest-PD and Nataf operations."""

import numpy as np
import pytest

import pystra as ra
from pystra.dependence.correlation import compute_modified_correlation_matrix


def test_valid_matrix_is_held_read_only():
    C = ra.CorrelationMatrix([[1.0, 0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(C.matrix, [[1.0, 0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(np.asarray(C), C.matrix)
    assert C[0, 1] == 0.5
    with pytest.raises(ValueError):
        C.matrix[0, 1] = 0.2
    with pytest.raises(TypeError):
        C[0, 1] = 0.2


@pytest.mark.parametrize(
    "matrix, reason",
    [
        ([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]], "square"),
        ([[1.0, np.nan], [np.nan, 1.0]], "finite"),
        ([[1.0, 0.5], [0.4, 1.0]], "symmetric"),
        ([[2.0, 0.5], [0.5, 1.0]], "unit diagonal"),
        ([[1.0, 1.2], [1.2, 1.0]], "positive definite"),
        ([[1.0, 1.0], [1.0, 1.0]], "positive definite"),
    ],
)
def test_invalid_matrices_are_rejected(matrix, reason):
    with pytest.raises(ra.ModelError, match=reason):
        ra.CorrelationMatrix(matrix)


def test_set_correlation_validates_arrays():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    with pytest.raises(ra.ModelError, match="positive definite"):
        model.set_correlation([[1.0, 1.5], [1.5, 1.0]])
    model.set_correlation([[1.0, 0.3], [0.3, 1.0]])
    np.testing.assert_array_equal(model.get_correlation(), [[1.0, 0.3], [0.3, 1.0]])


def test_cholesky_factor_reproduces_the_matrix():
    C = ra.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    L = C.cholesky()
    np.testing.assert_allclose(L @ L.T, C.matrix, atol=1e-14)
    np.testing.assert_array_equal(L, np.tril(L))


def test_nearest_positive_definite_reproduces_highams_example():
    # Higham (2002), Section 1: the nearest correlation matrix to this
    # indefinite unit-diagonal matrix, to four decimal places.
    C = ra.CorrelationMatrix.nearest_positive_definite(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    expected = [[1.0, 0.7607, 0.1573], [0.7607, 1.0, 0.7607], [0.1573, 0.7607, 1.0]]
    np.testing.assert_allclose(C.matrix, expected, atol=1e-4)
    assert np.linalg.eigvalsh(C.matrix).min() > 0


def test_nearest_positive_definite_keeps_a_valid_matrix():
    R = [[1.0, 0.3], [0.3, 1.0]]
    np.testing.assert_allclose(
        ra.CorrelationMatrix.nearest_positive_definite(R).matrix, R, atol=1e-12
    )


def test_nataf_matches_the_model_computation():
    C = ra.CorrelationMatrix([[1.0, 0.5], [0.5, 1.0]])
    normal = ra.StochasticModel()
    normal.add_variable(ra.Normal("X", 0, 1))
    normal.add_variable(ra.Normal("Y", 0, 1))
    np.testing.assert_allclose(C.nataf(normal).matrix, C.matrix, atol=1e-4)
    skewed = ra.StochasticModel()
    skewed.add_variable(ra.Lognormal("X", 10, 3))
    skewed.add_variable(ra.Gumbel("Y", 5, 2))
    skewed.set_correlation(C)
    np.testing.assert_array_equal(
        C.nataf(skewed).matrix, compute_modified_correlation_matrix(skewed)
    )
    assert C.nataf(skewed)[0, 1] != 0.5
    with pytest.raises(ra.ModelError, match="random variables"):
        ra.CorrelationMatrix(np.eye(3)).nataf(skewed)
