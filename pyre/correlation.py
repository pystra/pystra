#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from integration import *

class CorrelationMatrix(object):
  """
  """
  
  def __init__(self, matrix=None):
    """
    
    
    Arguments:
      - `matrix`:
    """
    self.matrix = matrix
    self.mu = None
    self.sigma = None
    self.p1 = None
    self.p2 = None
    self.p3 = None
    self.p4 = None

  def __repr__(self):
        return repr(self.matrix)

  def __getitem__(self, key):
    return self.matrix[key]

  def __setitem__(self, key, item):
    self.matrix[key] = item

  def getMatrix(self):
    return self.matrix


def getModifiedCorrelationMatrix(stochastic_model):
  marg = stochastic_model.getMarginalDistributions()
  R = stochastic_model.getCorrelation()
  nvr = len(marg)
  n,m = np.shape(R)
  Ro = np.eye(n,m)

  for i in range(nvr):
    for j in range(i):
      rho = R[i][j]
      if rho != 0 or flag_sens:

        margi = marg[i]
        margj = marg[j]

        zmax = 6

        if np.absolute(rho) > 0.9995:
          nIP = 1024
        elif np.absolute(rho) > 0.998:
          nIP = 512
        elif np.absolute(rho) > 0.992:
          nIP = 256
        elif np.absolute(rho) > 0.97:
          nIP = 128
        elif np.absolute(rho) > 0.9:
          nIP = 64
        else:
          nIP = 32

        Z1,Z2,X1,X2,WIP,detJ = zi_and_xi(margi,margj,zmax,nIP)

      if rho != 0:
        par = opt.fmin(absoluteIntegralValue, rho, args =(rho,margi,margj,Z1,Z2,X1,X2,WIP,detJ),disp=False)
        rho0 = par[0]
      else:
        rho0 = 0

      Ro[i][j] = rho0

  Ro = Ro + np.transpose(np.tril(Ro,-1))

  # Some parts are missing !!!

  return Ro

def absoluteIntegralValue(rho0,*args):
  """Absolute rho-integral value
  Compute the absolute value of the bi-folded rho-integral by 2D numerical integration
  """
  rho_target,margi,margj,Z1,Z2,X1,X2,WIP,detJ = args

  f = np.absolute( rho_target - rho_integral(rho0,margi,margj,Z1,Z2,X1,X2,WIP,detJ) )
  return f
