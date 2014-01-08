#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class Lognormal(Distribution):
  """Lognormal distribution

  :Arguments:
    - name (str):         Name of the random variable
    - mean (float):       Mean or lamb
    - stdv (float):       Standard deviation or zeta\n
    - input_type (any):   Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 2
    self.distribution = {2:'Lognormal'}
    self.name = name
    self.mean = mean
    self.stdv = stdv
    self.input_type = input_type
    mean,stdv,p1,p2,p3,p4 = self.setMarginalDistribution()
    Distribution.__init__(self,name,self.type,mean,stdv,startpoint,p1,p2,p3,p4,input_type)

  def setMarginalDistribution(self):
    """Compute the marginal distribution
    """
    if self.input_type == None:
      mean = self.mean
      stdv = self.stdv
      cov = stdv*mean**(-1)
      zeta = (np.log(1+cov**2))**0.5
      lamb = np.log(mean) - 0.5*zeta**2
      p3 = 0
      p4 = 0
    else:
      lamb = self.mean
      zeta = self.stdv
      mean = np.exp(lamb+0.5*(zeta**2))
      stdv = np.exp(lamb+0.5*(zeta**2)) * (np.exp(zeta**2)-1)**0.5
      p3 = 0
      p4 = 0
    return mean, stdv, lamb, zeta,p3,p4

  @classmethod
  def pdf(self,x,lamb=None,zeta=None,var_3=None,var_4=None):
    """probability density function
    """
    p = 1*( np.sqrt(2*np.pi) * zeta * x )**(-1) * np.exp( -0.5 * ((np.log(x)-lamb)*zeta**(-1))**2 )
    return p

  @classmethod
  def cdf(self,x,lamb=None,zeta=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    z = ( np.log(x) - lamb ) * zeta**(-1)
    P = 0.5+math.erf(z*np.sqrt(2)**(-1))*2**(-1)
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      lamb = marg.getP1()
      zeta = marg.getP2()
      x[i] = np.exp(u[i] * zeta + lamb)
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] =  ( np.log(x[i]) - marg.getP1() ) * marg.getP2()**(-1)
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      ksi = np.sqrt( np.log( 1 + ( marg.getStdv() * (marg.getMean())**(-1) )**2 ) )
      J[i][i] = 1 * ( ksi * x[i] )**(-1)
    return J
