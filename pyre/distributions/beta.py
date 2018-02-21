#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec
import scipy.stats as ss

from .distribution import *
from .normal import *

class Beta(Distribution):
  """Beta distribution

  :Attributes:
    - name (str):   Name of the random variable\n
    - mean (float): Mean or q\n
    - stdv (float): Standard deviation or r\n
    - a (float):    Lower boundary\n
    - b (float):    Uper boundary\n
    - input_type (any): Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n
  """

  def __init__(self, name,mean,stdv,a=0,b=1,input_type=None,startpoint=None):
    self.type = 7
    self.distribution = {7:'Beta'}
    self.name = name
    self.mean = mean
    self.stdv = stdv
    self.a = a
    self.b = b
    self.input_type = input_type
    mean,stdv,p1,p2,p3,p4 = self.setMarginalDistribution()
    Distribution.__init__(self,name,self.type,mean,stdv,startpoint,p1,p2,p3,p4,input_type)

  def setMarginalDistribution(self):
    """Compute the marginal distribution
    """
    if self.input_type == None:
      mean = self.mean
      stdv = self.stdv
      a    = self.a
      b    = self.b
      parameter_guess = 1
      par = opt.fmin(beta_parameter, parameter_guess, args =(a,b,mean,stdv),disp=False)
      q = par[0]
      r = q*(b-a)*(mean-a)**(-1) - q
    else:
      q = self.mean
      r = self.stdv
      a = self.a
      b = self.b
      # mean = a + q*(b-a)*(q+r)**(-1)
      # stdv = ((b-a)*(q+r)**(-1))*(q*r*(q+r+1)**(-1))**0.5
      mean = ss.beta(q, r, loc=a, scale=b-a).mean
      stdv = ss.beta(q, r, loc=a, scale=b-a).std
    return mean, stdv, q, r,a,b

  @classmethod
  def pdf(self,x,q=None,r=None,a=None,b=None):
    """probability density function
    """
    #p = (x-a)**(q-1) * (b-x)**(r-1) * ( (math.gamma(q)*math.gamma(r) *(math.gamma(q+r))**(-1)) * (b-a)**(q+r-1) )**(-1)
    p = ss.beta.pdf(x, q, r, loc=a, scale=b-a)
    return p

  @classmethod
  def cdf(self,x,q=None,r=None,a=None,b=None):
    """cumulative distribution function
    """
    # x01 = (x-a) * (b-a)**(-1)
    # P = spec.betainc(q, r, x01)
    P = ss.beta.cdf(x, q, r, loc=a, scale=b-a)
    return P

  @classmethod
  def u_to_x(self, u,marg,x=None):
    """Transformation from u to x
    """
    if x == None:
      x = np.zeros(len(u))
    for i in range(len(u)):
      q = marg.getP1()
      r = marg.getP2()
      a = marg.getP3()
      b = marg.getP4()
      # mean = marg.getMean()
      # normal_val = Normal.cdf(u[i],0,1)
      # par = opt.fminbound(zero_beta, 0,1, args =(q,r,normal_val),disp=False)
      # x01 = par
      # x[i] = a+x01*(b-a)
      x[i] = ss.beta.ppf(x, q, r, loc=a, scale=b-a)
    return x

  @classmethod
  def x_to_u(self, x, marg, u=None):
    """Transformation from x to u
    """
    if u == None:
      u = np.zeros(len(x))
    for i in range(len(x)):
      u[i] = Normal.inv_cdf( Beta.cdf(x[i],marg.getP1(),marg.getP2(),marg.getP3(),marg.getP4()) )
    return u

  @classmethod
  def jacobian(self,u,x,marg,J=None):
    """Compute the Jacobian
    """
    if J == None:
      J = np.zeros((len(marg),len(marg)))
    for i in range(len(marg)):
      pdf1 = Beta.pdf(x[i],marg.getP1(),marg.getP2(),marg.getP3(),marg.getP4())
      pdf2 = Normal.pdf(u[i],0,1)
      J[i][i] = pdf1*(pdf2)**(-1)
    return J



def beta_parameter(q,*args):
  a,b,mean,stdv = args
  r = (b-mean)*(mean-a)**(-1)*q
  f = np.absolute( ((b-a)*(q+r)**(-1))*(q*r*(q+r+1)**(-1))**0.5 - stdv )
  return f

def zero_beta(x,*args):
  q,r,normal_val = args
  zero_beta = np.absolute(spec.betainc(q, r, x) - normal_val )
  return zero_beta
