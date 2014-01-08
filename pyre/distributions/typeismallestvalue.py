#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class TypeIsmallestValue(Distribution):
  """Type I smallest value distribution

  :Attributes:
    - name (str):   Name of the random variable\n
    - mean (float): Mean or u_1\n
    - stdv (float): Standard deviation or a_1\n
    - input_type (any): Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 12
    self.distribution = {12:'TypeIsmallestValue'}
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
      a_1 = np.pi*(stdv*np.sqrt(6))**(-1)
      u_1 = mean + (0.5772156649*stdv*np.sqrt(6))*(np.pi)**(-1)
      p3 = 0
      p4 = 0
    else:
      u_1 = self.mean
      a_1 = self.stdv
      mean = u_1 - 0.5772156649*a_1**(-1)
      stdv = np.pi*(a_1*6**0.5)**(-1)
      p3 = 0
      p4 = 0
    return mean, stdv, u_1, a_1,p3,p4

  @classmethod
  def pdf(self,x,u_1=None,a_1=None,var_3=None,var_4=None):
    """probability density function
    """
    p = a_1 * np.exp( a_1*(x-u_1) - np.exp(a_1*(x-u_1)) )
    return p

  @classmethod
  def cdf(self,x,u_1=None,a_1=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    P = 1 - np.exp( -np.exp( a_1*(x-u_1) ) )
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      u_1 = marg.getP1()
      a_1 = marg.getP2()
      x[i] = u_1 + (1*a_1**(-1)) * np.log( np.log( 1 * ( 1 - Normal.cdf(u[i],0,1) )**(-1) ) )
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( TypeIsmallestValue.cdf(x[i],marg.getP1(),marg.getP2()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = TypeIsmallestValue.pdf(x[i],marg.getP1(),marg.getP2())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*(pdf2)**(-1)
    return J
