#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from distributions import *
from correlation import *
from collections import OrderedDict

class StochasticModel(object):
  """Stochastic model"""

  def __init__(self):
    self.variables = OrderedDict() # use ordered dictionary to make sure that the order corresponds to the correlation matrix
    self.names = []
    self.marg = []
    self.correlation = None
    self.Ro = None
    self.Lo = None
    self.iLo = None
    self.call_function = 0
  
  def addVariable(self, obj):
    """
    add stochastic variable
    """
    if not isinstance(obj, Distribution):
      raise Exception('input is not of type %s'%type(Distribution))
    if obj.getName() in self.names:
      raise Exception('variable name "%s" already exists'%obj.getName())
    # append the variable name
    self.names.append(obj.getName())
    # append marginal distribution
    self.marg.append(obj.getMarginalDistribution())
    # append the Distribution object to the variables (ordered) dictionary
    self.variables[obj.getName()] = obj
    # update the default correlation matrix, in accordance with the number of variables
    self.correlation = np.eye(len(self.marg))
    
  def getNames(self):
    return self.names

  def getLenMarginalDistributions(self):
    return len(self.marg)

  def getMarginalDistributions(self):
    return self.marg

  def setMarginalDistributions(self,marg):
    self.marg = marg

  def setCorrelation(self, obj):
    self.correlation = np.array(obj.getMatrix())

  def getCorrelation(self):
    return self.correlation

  def setModifiedCorrelation(self,correlation):
    self.Ro = correlation

  def getModifiedCorrelation(self):
    return self.Ro

  def setLowerTriangularMatrix(self,matrix):
    self.Lo = matrix

  def getLowerTriangularMatrix(self):
    return self.Lo

  def setInvLowerTriangularMatrix(self,matrix):
    self.iLo = matrix

  def getInvLowerTriangularMatrix(self):
    return self.iLo

  def addCallFunction(self,add):
    self.call_function += add

  def getCallFunction(self):
    return self.call_function



class AnalysisOptions(object):
  """Options

  Options for the structural reliability analysis. 
  """

  def __init__(self):

    self.transf_type = 3
    """Type of joint distribution

    :Type:
      - 1: jointly normal (no longer supported)\n
      - 2: independent non-normal (no longer supported)\n
      - 3: Nataf joint distribution (only available option)
    """

    self.Ro_method   = 1
    """Method for computation of the modified Nataf correlation matrix

    :Methods:
      - 0: use of approximations from ADK's paper (no longer supported)\n
      - 1: exact, solved numerically
    """

    self.flag_sens   = True
    """ Flag for computation of sensitivities

    w.r.t. means, standard deviations, parameters and correlation coefficients

    :Flag:
      - 1: all sensitivities assessed,\n
      - 0: no sensitivities assessment
    """

    self.print_output = True
    """Print comments during calculation

    :Values:
      - True: FERUM interactive mode,\n
      - False: FERUM silent mode
    """

    self.multi_proc = 1
    """ Amount of g-calls

    1: block_size g-calls sent simultaneously
       - gfunbasic.m is used and a vectorized version of gfundata.expression
         is available. The number of g-calls sent simultaneously (block_size)
         depends on the memory available on the computer running FERUM.
       - gfunxxx.m user-specific g-function is used and able to handle
         block_size computations            sent simultaneously, on a cluster
         of PCs or any other multiprocessor computer platform.
    0: g-calls sent sequentially
    """

    self.block_size = 1000
    """ Block size

    Number of g-calls to be sent simultaneously
    """

    # FORM analysis options
    self.i_max = 100
    """Maximum number of iterations allowed in the search algorithm"""

    self.e1 = 0.001
    """Tolerance on how close design point is to limit-state surface"""

    self.e2 = 0.001
    """Tolerance on how accurately the gradient points towards the origin"""

    self.step_size = 0
    """ Step size

    0: step size by Armijo rule, otherwise: given value is the step size
    """

    self.Recorded_u = True
    # 0: u-vector not recorded at all iterations,
    # 1: u-vector recorded at all iterations
    self.Recorded_x = True
    # 0: x-vector not recorded at all iterations,
    # 1: x-vector recorded at all iterations

    # FORM, SORM analysis options
    self.differentation_modus = 'ffd'
    """ Kind of differentiation

    :Type:
      - 'ddm': direct differentiation,\n
      - 'ffd': forward finite difference
    """

    self.ffdpara = 1000
    """ Parameter for computation

    Parameter for computation of FFD estimates of gradients - Perturbation =
    stdv/analysisopt.ffdpara\n

    :Values:
      - 1000 for basic limit-state functions,\n
      -  50 for FE-based limit-state functions
    """

    self.ffdpara_thetag = 1000
    # Parameter for computation of FFD estimates of dbeta_dthetag
    # perturbation = thetag/analysisopt.ffdpara_thetag if thetag ~= 0
    # or 1/analysisopt.ffdpara_thetag if thetag == 0;
    # Recommended values: 1000 for basic limit-state functions,
    #100 for FE-based limit-state functions

    # Simulation analysis (MC,IS,DS,SS) and distribution analysis options
    self.samples = 100000
    """Number of samples (MC,IS)

    Number of samples per subset step (SS) or number of directions (DS)
    """

    self.random_generator = 0
    """Kind of Random generator

    :Type:
      - 0: default rand matlab function,\n
      - 1: Mersenne Twister (to be preferred)
    """

    # Simulation analysis (MC, IS) and distribution analysis options
    self.sim_point = 'origin'
    """Start point for the simulation

    :Start:
      - 'dspt': design point,\n
      - 'origin': origin in standard normal space (simulation analysis)
    """

    self.stdv_sim = 1
    """Standard deviation of sampling distribution in simulation analysis"""

    # Simulation analysis (MC, IS)
    self.target_cov = 0.05
    """ Target coefficient of variation for failure probability"""

    # Bins of the histogram
    self.bins = None
    """Amount on bins for the histogram"""

  # getter
  def printOutput(self):
    return self.print_output

  def getFlagSens(self):
    return self.flag_sens

  def getMultiProc(self):
    return self.multi_proc

  def getBlockSize(self):
    return self.block_size

  def getImax(self):
    return self.i_max

  def getE1(self):
    return self.e1

  def getE2(self):
    return self.e2

  def getStepSize(self):
    return self.step_size

  def getDifferentationModus(self):
    return self.differentation_modus

  def getffdpara(self):
    return self.ffdpara

  def getBins(self):
    if self.bins != None:
      return self.bins
    else:
      bins = np.ceil(4*np.sqrt(np.sqrt(self.samples)))
      self.bins = bins
      return self.bins

  def getSamples(self):
    return self.samples

  def getRandomGenerator(self):
    return self.random_generator

  def getSimulationPoint(self):
    return self.sim_point

  def getSimulationStdv(self):
    return self.stdv_sim

  def getSimulationCov(self):
    return self.target_cov

  # setter
  def printResults(self,tof):
    self.print_output = tof

  def setMultiProc(self,multi_proc):
    self.multi_proc = multi_proc

  def setBlockSize(self,block_size):
    self.block_size = block_size

  def setImax(self,i_max):
    self.i_max = i_max

  def setE1(self,e1):
    self.e1 = e1

  def setE2(self,e2):
    self.e2 = e2

  def setStepSize(self,step_size):
    self.step_size = step_size

  def setDifferentationModus(self,differentation_modus):
    self.differentation_modus = differentation_modus

  def setffdpara(self,ffdpara):
    self.ffdpara = ffdpara

  def setBins(self,bins):
    self.bins = bins

  def setSamples(self,samples):
    self.samples = samples


class LimitState(object):
  """Limit state"""

  def __init__(self, expression=None):
    self.evaluator  = 'basic'
    """Type of limit-state function evaluator:

    :Args:
      basic: the limit-state function is defined by means of an analytical expression or a Python function.
    """

    # Do no change this field!
    self.type       = 'expression'

    self.expression = expression
    """Expression of the limit-state function"""

    self.flag_sens  = True
    """Flag for computation of sensitivities

    w.r.t. Tag parameters of the limit-state function

    :Tag:
      - 1: all sensitivities assessed,\n
      - 0: no sensitivities assessment\n
    """

  def getEvaluator(self):
    return self.evaluator

  def getExpression(self):
    return self.expression

  def setExpression(self, expression):
    self.expression = expression
#    inlist = False
#    for obj in gc.get_objects():
#      if isinstance(obj, LimitStateFunction):
#        self.expression = obj.getExpression()
#        inlist = True
#        break
#    if not inlist:
#      print 'Attention: No limit state function is defined'


class LimitStateFunction(object):

  def __init__(self, expression):
    self.expression = expression

  def __repr__(self):
    return self.expression

  def getExpression(self):
    return self.expression

