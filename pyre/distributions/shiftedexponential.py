#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class ShiftedExponential(Distribution):
  """Shifted exponential distribution

  :Attributes:
    - name (str):         Name of the random variable\n
    - mean (float):       Mean or lamb\n
    - stdv (float):       Standard deviation or x_zero\n
    - input_type (any):   Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 4
    self.distribution = {4:'ShiftedExponential'}
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
      x_zero= mean - stdv
      lamb= 1*stdv**(-1)
      p3 = 0
      p4 = 0
    else:
      lamb = self.mean
      x_zero = self.stdv
      mean = x_zero + 1*lamb**(-1)
      stdv = 1*lamb**(-1)
      p3 = 0
      p4 = 0
    return mean, stdv, lamb, x_zero,p3,p4

  @classmethod
  def pdf(self,x,lamb=None,x_zero=None,var_3=None,var_4=None):
    """probability density function
    """
    p = lamb * np.exp(-lamb*(x-x_zero))
    return p

  @classmethod
  def cdf(self,x,lamb=None,x_zero=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    P = 1 - np.exp( -lamb*(x-x_zero) )
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      lamb = marg.getP1()
      x_zero = marg.getP2()
      normal_val = Normal.cdf(u[i],0,1)
      x[i] = x_zero + 1*lamb**(-1) * np.log( 1 * ( 1 - normal_val )**(-1) )
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( ShiftedExponential.cdf(x[i],marg.getP1(),marg.getP2()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = ShiftedExponential.pdf(x[i],marg.getP1(),marg.getP2())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*(pdf2)**(-1)
    return J
