#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from .distribution import *
from .normal import *

class Uniform(Distribution):
  """Uniform distribution
  
  :Attributes:
    - name (str):   Name of the random variable\n
    - mean (float): Mean or a\n
    - stdv (float): Standard deviation or b\n
    - input_type (any): Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """
  
  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 6
    self.distribution = {6:'Uniform'}
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
      a = mean - 3**0.5 * stdv
      b = mean + 3**0.5 * stdv
      p3 = 0
      p4 = 0
    else:
      a = self.mean
      b = self.stdv
      mean = (a+b)*0.5
      stdv = (b-a)*(2*(3)**0.5)**(-1)
      p3 = 0
      p4 = 0
    return mean, stdv, a, b,p3,p4


  @classmethod
  def pdf(self,x,a=None,b=None,var_3=None,var_4=None):
    """probability density function
    """
    if len(x) == 1:
      p = 1 * (b-a)**(-1)
    else:
      p = []
      for i in range(len(x)):
        p.append(1 * (b-a)**(-1))

    return p

  @classmethod
  def cdf(self,x,a=None,b=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    P = (x-a) * (b-a)**(-1)
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      a = marg.getP1()
      b = marg.getP2()
      x[i] = a + (b-a) * Normal.cdf(u[i],0,1)
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( Uniform.cdf(x[i],marg.getP1(),marg.getP2()))
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = Uniform.pdf(x[i],marg.getP1(),marg.getP2())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*(pdf2)**(-1)
    return J
