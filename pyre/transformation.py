#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distributions import *

def z_to_x(z,marg):
  x = eval(getDistributionType(marg.getType())).u_to_x(z,marg)
  return x

def x_to_u(x,stochastic_model):
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  u = np.zeros(nrv)
  iLo = stochastic_model.getInvLowerTriangularMatrix()
  for i in range(nrv):
    u[i] = eval(getDistributionType(marg[i].getType())).x_to_u([x[i]],marg[i])

  u = np.dot(iLo,u)
  return u

def u_to_x(u,stochastic_model):
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  x = np.zeros(nrv)
  Lo = stochastic_model.getLowerTriangularMatrix()
  u = np.dot(Lo,u)

  for i in range(nrv):
    x[i] = eval(getDistributionType(marg[i].getType())).u_to_x([u[i]],marg[i])
  return x

def jacobian(u,x,stochastic_model):
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

