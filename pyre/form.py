#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import os
import numpy as np
import math
import scipy.optimize as opt
import scipy.special as spec

from model import *
from correlation import *
from distributions import *
from cholesky import *
from limitstate import *
from stepsize import *

class Form(object):
  """
  """

  def __init__(self,analysis_options=None,limit_state=None,stochastic_model=None):
    """
    
    """
    # The stochastic model
    if stochastic_model == None:
      self.model = StochasticModel()
    else:
      self.model = stochastic_model

    # Options for the calculation
    if analysis_options == None:
      self.options = AnalysisOptions()
    else:
      self.options = analysis_options

    # The limit state function
    if stochastic_model == None:
      self.limitstate = LimitState()
    else:
      self.limitstate = limit_state

    self.i = None
    self.u = None
    self.x = None
    self.J = None
    self.G = None
    self.Go = None
    self.gradient = None
    self.alpha = None
    self.gamma = None
    self.d = None
    self.step = None
    self.beta = None
    self.Pf = None
    
    # Computation of modified correlation matrix R0
    computeModifiedCorrelationMatrix(self)

    # Cholesky decomposition
    computeCholeskyDecomposition(self)

    # Compute starting point for the algorithm
    self.computeStartingPoint()

    # Iterations
    # Set parameters for the iterative loop
    # Initialize counter
    i = 1
    # Convergence is achieved when convergence is set to True
    convergence = False

    # loope
    while not convergence:
      if self.options.printOutput():
        print '.......................................'
        print 'Now carrying out iteration number:',i

      # comoute Transformation from u to x space
      self.computeTransformation()

      # compute the Jacobian
      self.computeJacobian()

      # Evaluate limit-state function and its gradient
      self.computeLimitState()

      # Set scale parameter Go and inform about struct. resp.
      if i == 1:
        self.Go = self.G
        if self.options.printOutput():
          print 'Value of limit-state function in the first step:', self.G
      # Compute alpha vector
      self.computeAlpha()

      # Compute gamma vector
      self.computeGamma()

      # Check convergence

      condition1 = np.absolute(self.G*self.Go**(-1))<self.options.getE1()
      condition2 = np.linalg.norm(self.u-self.alpha.dot(self.u).dot(self.alpha)) < self.options.getE2()
      condition3 = i==self.options.getImax()

      if condition1 and condition2 or condition3:
        self.i = i
        convergence = True

      # space for some recording stuf

      # Take a step if convergence is not achieved
      if not convergence:
        # Determine search direction
        self.computeSearchDirection()

        # Determine step size
        self.computeStepSize()

        # Determine new trial point
        u_new = self.u + self.step * self.d

        # Prepare for a new round in the loop
        self.u = u_new[0]#np.transpose(u_new)
        i += 1
      #   print self.u
      # if i == 5:
      #    convergence = True

    # Compute beta value
    self.computeBeta()

    # Compute failure probability
    self.computeFailureProbability()

    # Show Results
    if self.options.printOutput():
      self.showResults()


  # def computeModifiedCorrelationMatrix(self):
  #   if self.options.printOutput():
  #     print '=================================================='
  #     print ''
  #     print '        RUNNING FORM RELIABILITY ANALYSIS'
  #     print ''
  #     print '=================================================='
  #     print ''
  #     print ' Computation of modified correlation matrix R0'
  #     print ' Takes some time if sensitivities are to be computed'
  #     print ' with gamma (3), beta (7) or chi-square (8)'
  #     print ' distributions.'
  #     print ' Please wait... (Ctrl+C breaks)'
  #     print ''
  #   # Compute corrected correlation coefficients
  #   Ro = getModifiedCorrelationMatrix(self.model)
  #   self.model.setModifiedCorrelation(Ro)
  #   #print self.model.getModifiedCorrelation()

  # def computeCholeskyDecomposition(self):
  #   Ro = self.model.getModifiedCorrelation()
  #   Lo, ierr = CholeskyDecomposition(Ro)
  #   if  ierr > 0:
  #     print 'Error: Cholesky decomposition',ierr

  #   self.model.setLowerTriangularMatrix(Lo)
  #   iLo = np.linalg.inv(Lo)
  #   self.model.setInvLowerTriangularMatrix(iLo)

  def computeStartingPoint(self):
    x = np.array([])
    marg = self.model.getMarginalDistributions()
    for i in range(len(marg)):
      x = np.append(x,marg[i].getStartPoint())
    self.u = x_to_u(x,self.model)

  def computeTransformation(self):
    self.x = np.transpose([u_to_x(self.u,self.model)]);

  def computeJacobian(self):
    J_u_x = jacobian(self.u,self.x,self.model);
    J_x_u = np.linalg.inv(J_u_x)
    self.J = J_x_u

  def computeLimitState(self):
    G, gradient = evaluateLimitState(self.x,self.model,self.options,self.limitstate)
    self.G = G
    self.gradient = np.dot(np.transpose(gradient),self.J)

  def computeAlpha(self):
    self.alpha = -self.gradient * np.linalg.norm(self.gradient)**(-1)

  def computeGamma(self):
    self.gamma = np.diag(np.diag(np.sqrt(np.dot(self.J,np.transpose(self.J)))))
    # Importance vector gamma
    matmult = np.dot(np.dot(self.alpha,self.J),self.gamma)
    importance_vector_gamma = (matmult*np.linalg.norm(matmult)**(-1))

  def computeSearchDirection(self):
    self.d = ( self.G * np.linalg.norm(self.gradient)**(-1) + self.alpha.dot(self.u) ) * self.alpha - self.u

  def computeStepSize(self):
    if self.options.getStepSize() == 0:
      self.step = getStepSize(self.G,self.gradient,self.u,self.d,self.model,self.options,self.limitstate)
    else:
      self.step = self.options.getStepSize()

  def computeBeta(self):
    self.beta = np.dot(self.alpha,self.u)

  def computeFailureProbability(self):
    self.Pf = Normal.cdf(-self.beta,0,1)

  def showResults(self):
    print ''
    print '=================================================='
    print ''
    print ' RESULTS FROM RUNNING FORM RELIABILITY ANALYSIS'
    print ''
    print ' Number of iterations:     ',self.i
    print ' Reliability index beta:   ',self.beta[0]
    print ' Failure probability:      ',self.Pf
    print ' Number of calls to the limit-state function:',self.model.getCallFunction()
    print ''
    print '=================================================='
    print ''



  def getBeta(self):
    return self.beta[0]

  def getFailure(self):
    return self.Pf

  def getDesignPoint(self):
    return self.u