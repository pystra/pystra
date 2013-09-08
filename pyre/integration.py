#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
#from alt import *
from transformation import *
from quadrature import *

import numpy as np

def rho_integral(rho0,margi,margj,Z1,Z2,X1,X2,WIP,detJ):
  """Integral for rho0"""
  PHI2 = 1*(2*np.pi*np.sqrt(1-rho0**2))**(-1) *np.exp( -1*(2*(1-rho0**2))**(-1) * ( Z1**2 - 2*rho0*Z1*Z2 + Z2**2) )
  rho = np.sum ( np.sum ( (X1-margi[1])*(margi[2])**(-1) * (X2-margj[1])*(margj[2])**(-1) * PHI2 * detJ * WIP ) )
  return rho

def zi_and_xi(margi,margj,zmax,nIP):
  """Values for the Gauss integration"""
  # Computes z1, z2 and x1, x2 values for Gauss integration - Vectorized version

  # Integration limits ( should be -infinity, +infinity on both axes, in theory )
  zmin = -zmax

  # Determinant of the jacobian of the transformation between [z1max,z1min]x[z2max,z2min] and [-1,1]x[-1,1]
  detJ = (zmax-zmin)**2*4**(-1)

  # Get integration points and weight in [-1,1], nIP is the number of integration pts
  xIP,wIP = quadratureRule(nIP)

  # Transform integration points coordinates from [-1,1] to [zmax,zmin]
  z1 = zmin * np.ones(len(xIP)) + (zmax-zmin) * ( xIP + np.ones(len(xIP)) ) * 2**(-1)
  z2 = z1

  x1 = z_to_x(z1,margi)
  x2 = z_to_x(z2,margj)

  v1 = np.ones(nIP)
  v2 = np.transpose([v1])

  Z1 = np.dot(np.transpose([z1]),[v1])
  Z2 = np.dot(v2,[z2])
  X1 = np.dot(np.transpose([x1]),[v1])
  X2 = np.dot(v2,[x2])
  WIP = np.dot(np.transpose([wIP]),[wIP])

  return Z1,Z2,X1,X2,WIP,detJ

