#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

def computeCholeskyDecomposition(self):
  """Compute Cholesky Decomposition"""
  Ro = self.model.getModifiedCorrelation()
  Lo, ierr = CholeskyDecomposition(Ro)
  if  ierr > 0:
    print 'Error: Cholesky decomposition',ierr

  self.model.setLowerTriangularMatrix(Lo)
  iLo = np.linalg.inv(Lo)
  self.model.setInvLowerTriangularMatrix(iLo)

def CholeskyDecomposition(A):
  """Cholesky Decomposition

  The Cholesky decomposition of a Hermitian positive-definite matrix
  :math:`{\\bf A}` is a decomposition of the form

  .. math::

     \\mathbf{A = L L}^{*}

  where :math:`{\\bf L}` is a lower triangular matrix with positive diagonal
  entries, and :math:`{\\bf L}^*` denotes the conjugate transpose of
  :math:`{\\bf L}`. Every Hermitian positive-definite matrix (and thus also
  every real-valued symmetric positive-definite matrix) has a unique Cholesky
  decomposition.

  If the matrix :math:`{\\bf A}` is Hermitian and positive semi-definite, then
  it still has a decomposition of the form :math:`{\\bf A} = {\\bf LL}^*` if
  the diagonal entries of :math:`{\\bf L}` are allowed to be zero.

  When :math:`{\\bf A}` has real entries, :math:`{\\bf L}` has real entries as
  well.

  The Cholesky decomposition is unique when :math:`{\\bf A}` is positive
  definite; there is only one lower triangular matrix :math:`{\\bf L}` with
  strictly positive diagonal entries such that :math:`{\\bf A} = {\\bf
  LL}^*`. However, the decomposition need not be unique when :math:`{\\bf A}`
  is positive semidefinite.

  The converse holds trivially: if :math:`{\\bf A}` can be written as
  :math:`{\\bf LL}^*` for some invertible :math:`{\\bf L}`, lower triangular or
  otherwise, then A is Hermitian and positive definite.

  :Args:
    - A (mat): Hermitian positive-definite matrix :math:`{\\bf A}`

  :Returns:
    - Lo (mat): Returns a lower triangular matrix :math:`{\\bf L}` with positive diagonal entries

  """
  n,n = A.shape
  ierr = 0
  for k in range(n):
    if A[k][k] <= 0:
      ierr = k
      print 'Error: in Choleski decomposition - Matrix must be positive definite\n'
      break
    A[k][k] = np.sqrt(A[k][k])
    indx = range(k+1,n)
    for i in indx:
      A[i][k] = A[i][k] * A[k][k]**(-1)

    for j in range(k+1,n):
      indx = range(j,n)
      for i in indx:
        A[i][j] = A[i][j] - A[i][k]*A[j][k]
  Lo = np.tril(A)
  return Lo, ierr