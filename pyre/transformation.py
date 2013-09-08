#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distributions import *

def pdf(x,marg):
  """probability density function

  :Args:
    - x (vec): Vector with random values\n
    - marg (mat): Matrix with the marginal distributions

  :Returns:
    - p (mat): Returns a matrix with the pdfs of the marginal distributions
  """
  p = eval(getDistributionType(marg.getType())).pdf(x,marg.getP1(),marg.getP2(),marg.getP3(),marg.getP4())
  return p

def z_to_x(z,marg):
  """Transformation from z to x space"""
  x = eval(getDistributionType(marg.getType())).u_to_x(z,marg)
  return x

def x_to_u(x,stochastic_model):
  """Transformation from x to u space"""
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  u = np.zeros(nrv)
  iLo = stochastic_model.getInvLowerTriangularMatrix()
  for i in range(nrv):
    u[i] = eval(getDistributionType(marg[i].getType())).x_to_u([x[i]],marg[i])

  u = np.dot(iLo,u)
  return u

def u_to_x(u,stochastic_model):
  """Transformation from x to u space"""
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  x = np.zeros(nrv)
  Lo = stochastic_model.getLowerTriangularMatrix()
  u = np.dot(Lo,u)

  for i in range(nrv):
    x[i] = eval(getDistributionType(marg[i].getType())).u_to_x([u[i]],marg[i])
  return x

def jacobian(u,x,stochastic_model):
  """Jacobian for the transformation"""
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  Lo = stochastic_model.getLowerTriangularMatrix()
  iLo = stochastic_model.getInvLowerTriangularMatrix()
  u = np.dot(Lo,u)
  J_u_x = np.zeros((nrv,nrv))

  for i in range(nrv):
    J_u_x[i][i] = eval(getDistributionType(marg[i].getType())).jacobian([u[i]],[x[i]],marg[i])

  J_u_x = np.dot(iLo,J_u_x)
  return J_u_x

def getBins(samples):
  """Return an optimal amount of bins for a histogram

  :Returns:
    - bins (int): Returns amount on bins
  """
  bins = np.ceil(4*np.sqrt(np.sqrt(samples)))
  return bins