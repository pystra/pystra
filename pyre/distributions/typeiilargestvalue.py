#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class TypeIIlargestValue(Distribution):
  """Type II largest value distribution

  :Attributes:
    - name (str):   Name of the random variable\n
    - mean (float): Mean or u_n\n
    - stdv (float): Standard deviation or k\n
    - input_type (any): Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,input_type=None,startpoint=None):
    self.type = 13
    self.distribution = {13:'TypeIIlargestValue'}
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
      parameter_guess = [2.000001]#, 10**7]
      par = opt.fsolve(typIIlargest_parameter,parameter_guess, args =(mean,stdv))
      k = par[0]
      u_n = mean*(math.gamma(1-1*k**(-1)))**(-1)
      p3 = 0
      p4 = 0
    else:
      u_n = self.mean
      k   = self.stdv
      mean = u_n*math.gamma(1-1*k**(-1))
      stdv = u_n*(math.gamma(1-2*k**(-1))-(math.gamma(1-1*k**(-1)))**2)**0.5
      p3 = 0
      p4 = 0
    return mean, stdv, u_n, k,p3,p4

  @classmethod
  def pdf(self,x,u_n=None,k=None,var_3=None,var_4=None):
    """probability density function
    """
    p = k*u_n**(-1) * (u_n*x**(-1))**(k+1) * np.exp(-(u_n*x**(-1))**k)
    return p

  @classmethod
  def cdf(self,x,u_n=None,k=None,var_3=None,var_4=None):
    """cumulative distribution function
    """
    P = np.exp( -(u_n*x**(-1))**k )
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      u_n = marg.getP1()
      k = marg.getP2()
      x[i] = u_n * np.log( 1 *(Normal.cdf(u[i],0,1))**(-1) )**(-1*k**(-1))
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( TypeIIlargestValue.cdf(x[i],marg.getP1(),marg.getP2()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = TypeIIlargestValue.pdf(x[i],marg.getP1(),marg.getP2())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*(pdf2)**(-1)
    return J



def typIIlargest_parameter(x,*args):
  mean,stdv = args
  f = (math.gamma(1-2*x**(-1))-(math.gamma(1-1*x**(-1)))**2)**0.5 - (stdv*mean**(-1))*math.gamma(1-1*x**(-1))
  return f
