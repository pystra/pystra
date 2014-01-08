#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distribution import *
from normal import *

class TypeIIIsmallestValue(Distribution):
  """Type III smallest value distribution

  :Attributes:
    - name (str):   Name of the random variable\n
    - mean (float): Mean or u_1\n
    - stdv (float): Standard deviation or k\n
    - epsilon (float): Epsilon\n
    - input_type (any): Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,epsilon=0,input_type=None,startpoint=None):
    self.type = 14
    self.distribution = {14:'TypeIIIsmallestValue'}
    self.name = name
    self.mean = mean
    self.stdv = stdv
    self.epsilon = epsilon
    self.input_type = input_type
    mean,stdv,p1,p2,p3,p4 = self.setMarginalDistribution()
    Distribution.__init__(self,name,self.type,mean,stdv,startpoint,p1,p2,p3,p4,input_type)

  def setMarginalDistribution(self):
    """Compute the marginal distribution
    """
    if self.input_type == None:
      mean    = self.mean
      stdv    = self.stdv
      epsilon = self.epsilon
      meaneps = mean - epsilon
      parameter_guess = [0.1]#, 10**7]
      par = opt.fsolve(typIIIsmallest_parameter,parameter_guess, args =(meaneps,stdv))
      k = par[0]
      u_1 = meaneps*(math.gamma(1+1*k**(-1)))**(-1)+epsilon
      p4 = 0
    else:
      u_1     = self.mean
      k       = self.stdv
      epsilon = self.epsilon
      mean = epsilon + (u_1-epsilon)*math.gamma(1+1*k**(-1))
      stdv = (u_1-epsilon)*(math.gamma(1+2*k**(-1))-math.gamma(1+1*k**(-1))**2)**0.5
      p4 = 0
    return mean, stdv, u_1, k,epsilon,p4

  @classmethod
  def pdf(self,x,u_1=None,k=None,epsilon=None,var_4=None):
    """probability density function
    """
    p = k*(u_1-epsilon)**(-1) * ((x-epsilon)*(u_1-epsilon)**(-1))**(k-1) * np.exp(-((x-epsilon)*(u_1-epsilon)**(-1))**k)
    return p

  @classmethod
  def cdf(self,x,u_1=None,k=None,epsilon=None,var_4=None):
    """cumulative distribution function
    """
    P = 1 - np.exp( -( (x-epsilon) * (u_1-epsilon)**(-1) )**k )
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      u_1 = marg.getP1()
      k = marg.getP2()
      epsilon = marg.getP3()
      x[i] = epsilon + ( u_1 - epsilon ) * np.log( 1 * ( 1 - Normal.cdf(u[i],0,1) )**(-1) )**(1*k**(-1))
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( TypeIIIsmallestValue.cdf(x[i],marg.getP1(),marg.getP2(),marg.getP3()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = TypeIIIsmallestValue.pdf(x[i],marg.getP1(),marg.getP2(),marg.getP3())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*(pdf2)**(-1)
    return J



def typIIIsmallest_parameter(x,*args):
  meaneps,stdv = args
  f = (math.gamma(1+2*x**(-1))-(math.gamma(1+1*x**(-1)))**2)**0.5 - (stdv*meaneps**(-1))*math.gamma(1+1*x**(-1))
  return f
