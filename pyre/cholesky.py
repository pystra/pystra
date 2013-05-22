#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

def computeCholeskyDecomposition(self):
  Ro = self.model.getModifiedCorrelation()
  Lo, ierr = CholeskyDecomposition(Ro)
  if  ierr > 0:
    print 'Error: Cholesky decomposition',ierr

  self.model.setLowerTriangularMatrix(Lo)
  iLo = np.linalg.inv(Lo)
  self.model.setInvLowerTriangularMatrix(iLo)

def CholeskyDecomposition(A):
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

