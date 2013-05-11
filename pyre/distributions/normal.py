#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *

class Normal(Distribution):
  """Normal distribution
  
  :Attributes:
    - name (str):         Name of the random variable\n
    - mean (float):       Mean\n
    - stdv (float):       Standard deviation\n
    - input_type (any):   Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n 
  """

  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 1
    self.distirbution = {1:'Normal'}
    self.mean = mean
    self.stdv = stdv
    mean,stdv,p1,p2,p3,p4 = self.setMarginalDistribution()
    Distribution.__init__(self,name,self.type,mean,stdv,startpoint,p1,p2,p3,p4,input_type)


  def setMarginalDistribution(self):
    """Compute the marginal distribution  
    """
    return self.mean, self.stdv, self.mean, self.stdv,0,0

  @classmethod
  def pdf(self,x,mean=None,stdv=None,var_3=None,var_4=None):
    """probability density function
    """
    p = 1*( np.sqrt(2*np.pi) * stdv )**(-1) * np.exp( -0.5 * ((x-mean)*stdv**(-1))**2 )
    return p

  @classmethod
  def cdf(self,x,mean=None,stdv=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    P = 0.5+math.erf(((x-mean)*stdv**(-1))*np.sqrt(2)**(-1))*2**(-1)
    return P

  @classmethod
  def inv_cdf(self,P):
    """inverse cumulative distribution function
    """
    x = spec.erfinv(2*(P-0.5))*np.sqrt(2)
    return x

  @classmethod
  def u_to_x(self, u, marg, x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      x[i] = u[i] * marg.getStdv() + marg.getMean()
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] =  ( x[i] - marg.getMean() ) * marg.getStdv()**(-1)
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      J[i][i] = 1*(marg.getStdv())**(-1)
    return J

