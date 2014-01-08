#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class ChiSquare(Distribution):
  """Chi-Square distribution

  :Attributes:
    - name (str):   Name of the random variable\n
    - mean (float): Mean or nu\n
    - stdv (float): Standard deviation\n
    - input_type (any): Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv=None,input_type=None,startpoint=None):
    self.type = 8
    self.distribution = {8:'ChiSquare'}
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
      lamb = 0.5
      mean_test = lamb*stdv**2
      if mean*mean_test**(-1) < 0.95 or mean*mean_test**(-1) > 1.05:
        print 'Error when using Chi-square distribution. Mean and stdv should be given such that mean = 0.5*stdv.**2\n'
      nu = 2*((mean**2)*(stdv**2)**(-1))
      p2 = 0
      p3 = 0
      p4 = 0
    else:
      lamb = 0.5
      nu     = self.mean
      mean = nu*(2*lamb)**(-1)
      stdv = ((nu*0.5)**0.5)*lamb**(-1)
      p2 = 0
      p3 = 0
      p4 = 0
    return mean, stdv, nu, p2,p3,p4

  @classmethod
  def pdf(self,x,nu=None,var_2=None,var_3=None,var_4=None):
    """probability density function
    """
    lamb = 0.5
    k      = nu*0.5
    p = lamb * (lamb*x)**(k-1) * np.exp(-lamb*x) *(math.gamma(k))**(-1)
    return p

  @classmethod
  def cdf(self,x,nu=None,var_2=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    lamb = 0.5
    k      = nu*.5
    P = spec.gammainc(k,lamb*x)
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      lamb = 0.5
      nu = marg.getP1()
      k = nu*0.5
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
      u[i] = Normal.inv_cdf( ChiSquare.cdf(x[i],marg.getP1()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = ChiSquare.pdf(x[i],marg.getP1())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i]= pdf1*(pdf2)**(-1)
    return J



def zero_gamma(x,*args):
  k,lamb,normal_val = args
  zero_gamma = np.absolute(spec.gammainc(k,lamb*x) - normal_val )
  return zero_gamma
