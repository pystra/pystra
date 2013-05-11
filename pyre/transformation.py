#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from distributions import *

def z_to_x(z,marg):
  x = eval(getDistributionType(marg.getType())).u_to_x(z,marg)
  return x

def x_to_u(x,stochastic_model):
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  u = np.zeros(nrv)
  iLo = stochastic_model.getInvLowerTriangularMatrix()
  for i in range(nrv):
    u[i] = eval(getDistributionType(marg[i].getType())).x_to_u([x[i]],marg[i])

  u = np.dot(iLo,u)
  return u

def u_to_x(u,stochastic_model):
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  x = np.zeros(nrv)
  Lo = stochastic_model.getLowerTriangularMatrix()
  u = np.dot(Lo,u)
  
  for i in range(nrv):
    x[i] = eval(getDistributionType(marg[i].getType())).u_to_x([u[i]],marg[i])
  return x

def jacobian(u,x,stochastic_model):
  marg = stochastic_model.getMarginalDistributions()
  nrv = len(marg)
  Lo = stochastic_model.getLowerTriangularMatrix()
  iLo = stochastic_model.getInvLowerTriangularMatrix()
  u = np.dot(Lo,u)
  J_u_x = np.zeros((nrv,nrv))

  for i in range(nrv):
    J_u_x[i][i] = eval(getDistributionType(marg[i].getType())).jacobian([u[i]],[x[i]],marg[i])

  J_u_x = np.dot(iLo,J_u_x)
  return J_u_x

  
  
  
def z_to_x_a(z,marg,i):
  # Normal distribution
  if marg[i][0] == 1:
    #x = Normal.u_to_x(z,marg[i])
    x = z * marg[i][2] + marg[i][1]
    
  # Lognormal distribution
  if marg[i][0] == 2:
    lamb = marg[i][4]
    zeta = marg[i][5]
    x = np.exp(z * zeta + lamb)

  # Gamma distribution
  if marg[i][0] == 3:
    lamb = marg[i][4]
    k = marg[i][5]
    mean = marg[i][1]
    x = []
    for j in range(len(z)):
      normal_val = ferum_cdf(1,z[j],0,1)
      par = opt.fmin(zero_gamma, mean, args =(k,lamb,normal_val),disp=False)
      x.append(par[0])

  # Shifted exponential distribution
  if marg[i][0] == 4:
    lamb = marg[i][4]
    x_zero = marg[i][5]
    x = []
    normal_val = ferum_CDF(1,z,0,1)
    x = x_zero + 1*lamb**(-1) * np.log( 1 * ( 1 - normal_val )**(-1) )

  # Shifted Rayleigh distribution
  if marg[i][0] == 5:
    a = marg[i][4]
    x_zero = marg[i][5]
    x= x_zero + a * ( 2*np.log( 1 * (1-ferum_CDF(1,z,0,1))**(-1) ) ) **0.5

  # Uniform distribution
  if marg[i][0] == 6:
    a = marg[i][4]
    b = marg[i][5]
    x = a + (b-a) * ferum_CDF(1,z,0,1)

  # Beta distribution
  if marg[i][0] == 7:
    q = marg[i][4]
    r = marg[i][5]
    a = marg[i][6]
    b = marg[i][7]
    mean = marg[i][1]
    x = []
    for j in range(len(z)):
      normal_val = ferum_cdf(1,z[j],0,1)
      par = opt.fminbound(zero_beta, 0,1, args =(q,r,normal_val),disp=False)
      x01 = par
      x.append(a+x01*(b-a))

  # Chi-square distribution
  if marg[i][0] == 8:
    lamb = 0.5
    nu = marg[i][4]
    k = nu*0.5 
    mean = marg[i][1]
    x = []
    for j in range(len(z)):
      normal_val = ferum_cdf(1,z[j],0,1)
      par = opt.fmin(zero_gamma, mean, args =(k,lamb,normal_val),disp=False)
      x.append(par[0])

  # Type I largest value distribution ( same as Gumbel distribution )
  if marg[i][0] == 11:
    u_n = marg[i][4]
    a_n = marg[i][5]
    x = u_n - (1*a_n**(-1)) * np.log( np.log( 1 * (ferum_CDF(1,z,0,1))**(-1) ) )

  # Type I smallest value distribution
  if marg[i][0] == 12:
    u_1 = marg[i][4]
    a_1 = marg[i][5]
    x = u_1 + (1*a_1**(-1)) * np.log( np.log( 1 * ( 1 - ferum_CDF(1,z,0,1) )**(-1) ) )

  # Type II largest value distribution
  if marg[i][0] == 13:
    u_n = marg[i][4]
    k = marg[i][5]
    x = u_n * np.log( 1 *(ferum_CDF(1,z,0,1))**(-1) )**(-1*k**(-1))

  # Type III smallest value distribution
  if marg[i][0] == 14:
    u_1 = marg[i][4]
    k = marg[i][5]
    epsilon = marg[i][6]
    x = epsilon + ( u_1 - epsilon ) * np.log( 1 * ( 1 - ferum_CDF(1,z,0,1) )**(-1) )**(1*k**(-1))

  # Gumbel distribution ( same as type I largest value distribution )
  if marg[i][0] == 15:
    u_n = marg[i][4]
    a_n = marg[i][5]
    x = u_n - (1*a_n**(-1)) * np.log( np.log( 1 *(ferum_CDF(1,z,0,1))**(-1) ) )

  # Weibull distribution ( same as Type III smallest value distribution with epsilon = 0 )
  if marg[i][0] == 16:
    u_1 = marg[i][4]
    k = marg[i][5]
    x = u_1 * np.log( 1 * ( 1 - ferum_CDF(1,z,0,1) )**(-1) )**(1*k**(-1))
  return x

def ferum_CDF(type,x,var_1,var_2=None,var_3=None,var_4=None):
  P = np.array([])
  for i in range(len(x)):
    par =ferum_cdf(type,x[i],var_1,var_2,var_3,var_4)
    P = np.append(P,par)
  return P

def ferum_cdf(type,x,var_1,var_2=None,var_3=None,var_4=None):
  # Cumulative Density Function
  #
  #   P = ferum_cdf(type,x,varargin)
  #
  #   Evaluates the cumulative distribution function and returns the probability.
  #
  #   Output: - P           = cumulative density value 
  #   Input:  - type        = probability distribution type (1: normal, 2: lognormal, ...)
  #        - x           = 'Abscissa' value(s)
  #        - var_1 = parameter #1 of the random variable
  #        - var_2 = parameter #2 of the random variable (optional)
  #        - var_3 = parameter #3 of the random variable (optional)
  #        - var_4 = parameter #4 of the random variable (optional)
  if type == 1:  # Normal marginal distribution
    mean = var_1
    stdv = var_2
    P = 0.5+math.erf(((x-mean)*stdv**(-1))*np.sqrt(2)**(-1))*2**(-1)

  if type == 2:  # Lognormal marginal distribution
    lamb = var_1
    zeta   = var_2
    z = ( np.log(x) - lamb ) * zeta**(-1)
    P = 0.5+math.erf(z*np.sqrt(2)**(-1))*2**(-1)

  if type == 3:  # Gamma distribution
    lamb = var_1
    k      = var_2
    P = spec.gammainc(k,lamb*x)

  if type == 4:  # Shifted exponential distribution
    lamb = var_1
    x_zero = var_2
    P = 1 - np.exp( -lamb*(x-x_zero) )

  if type == 5:  # Shifted Rayleigh distribution
    a      = var_1
    x_zero = var_2
    P = 1 - np.exp( -0.5*( (x-x_zero)*a**(-1) )**2 )

  if type == 6:  # Uniform distribution
    a = var_1
    b = var_2
    P = (x-a) * (b-a)**(-1)

  if type == 7:  # Beta distribution
    q = var_1
    r = var_2
    a = var_3
    b = var_4
    x01 = (x-a) * (b-a)**(-1)
    P = spec.betainc(q, r, x01)

  if type == 8:  # Chi-square distribution
    nu     = var_1
    lamb = 0.5
    k      = nu*.5
    P = spec.gammainc(k,lamb*x)

  if type == 11:  # Type I largest value distribution ( same as Gumbel distribution )
    u_n = var_1
    a_n = var_2
    P = np.exp( -np.exp( -a_n*(x-u_n) ) )

  if type == 12:  # Type I smallest value distribution
    u_1 = var_1
    a_1 = var_2
    P = 1 - np.exp( -np.exp( a_1*(x-u_1) ) )

  if type == 13:  # Type II largest value distribution
    u_n = var_1
    k   = var_2
    P = np.exp( -(u_n*x**(-1))**k )

  if type == 14:  # Type III smallest value distribution
    u_1     = var_1
    k       = var_2
    epsilon = var_3
    P = 1 - np.exp( -( (x-epsilon) * (u_1-epsilon)**(-1) )**k )

  if type == 15:  # Gumbel distribution ( same as type I largest value distribution )
    u_n = var_1
    a_n = var_2
    P = np.exp( -np.exp( -a_n*(x-u_n) ) )

  if type == 16:  # Weibull distribution ( same as Type III smallest value distribution with epsilon = 0 )
    u_1 = var_1
    k   = var_2
    P = 1 - np.exp( -(x*u_1**(-1))**k )

  return P
