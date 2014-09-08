#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class Gumbel(Distribution):
  """Gumbel distribution

  :Attributes:
    - name (str):   Name of the random variable\n
    - mean (float): Mean or u_n\n
    - stdv (float): Standard deviation or a_n\n
    - input_type (any): Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 15
    self.distribution = {15:'Gumbel'}
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
      a_n = np.pi*(stdv*np.sqrt(6))**(-1)
      u_n = mean - (0.5772156649*stdv*np.sqrt(6))*(np.pi)**(-1)
      p3 = 0
      p4 = 0
    else:
      u_n = self.mean
      a_n = self.stdv
      mean = u_n + 0.5772156649*a_n**(-1)
      stdv = np.pi*(a_n*6**0.5)**(-1)
      p3 = 0
      p4 = 0
    return mean, stdv, u_n, a_n,p3,p4

  @classmethod
  def pdf(self,x,u_n=None,a_n=None,var_3=None,var_4=None):
    """probability density function
    """
    p = a_n * np.exp( -a_n*(x-u_n) - np.exp(-a_n*(x-u_n)) )
    return p

  @classmethod
  def cdf(self,x,u_n=None,a_n=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    P = np.exp( -np.exp( -a_n*(x-u_n) ) )
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      u_n = marg.getP1()
      a_n = marg.getP2()
      x[i] = u_n - (1*a_n**(-1)) * np.log( np.log( 1 * (Normal.cdf(u[i],0,1))**(-1) ) )
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( Gumbel.cdf(x[i],marg.getP1(),marg.getP2()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = Gumbel.pdf(x[i],marg.getP1(),marg.getP2())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*(pdf2)**(-1)
    return J
