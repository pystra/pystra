#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class Gamma(Distribution):
  """Gamma distribution

  :Attributes:
    - name (str):         Name of the random variable\n
    - mean (float):       Mean or lamb\n
    - stdv (float):       Standard deviation or k\n
    - input_type (any):   Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 3
    self.distribution = {3:'Gamma'}
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
      mean = self.mean
      stdv = self.stdv
      lamb = mean * (stdv**2)**(-1)
      k = mean**2 * (stdv**2)**(-1)
      p3 = 0
      p4 = 0
    else:
      lamb = self.mean
      k = self.stdv
      mean = k*lamb**(-1)
      stdv = (k**0.5)*lamb**(-1)
      p3 = 0
      p4 = 0
    return mean, stdv, lamb, k,p3,p4

  @classmethod
  def pdf(self,x,lamb=None,k=None,var_3=None,var_4=None):
    """probability density function
    """
    p = lamb * (lamb*x)**(k-1) *(spec.gamma(k))**(-1) * np.exp(-lamb*x)
    return p

  @classmethod
  def cdf(self,x,lamb=None,k=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    P = spec.gammainc(k,lamb*x)
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      lamb = marg.getP1()
      k = marg.getP2()
      mean = marg.getMean()
      normal_val = Normal.cdf(u[i],0,1)
      par = opt.fmin(zero_gamma, mean, args =(k,lamb,normal_val),disp=False)
      x[i] = par[0]
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( Gamma.cdf(x[i],marg.getP1(),marg.getP2()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = Gamma.pdf(x[i],marg.getP1(),marg.getP2())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*pdf2**(-1)
    return J



def zero_gamma(x,*args):
  k,lamb,normal_val = args
  zero_gamma = np.absolute(spec.gammainc(k,lamb*x) - normal_val )
  return zero_gamma
