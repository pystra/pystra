#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from limitstate import *
from transformation import *

def getStepSize(G,gradient,u,d,stochastic_model,analysis_options,limit_state):
  """Return the step size for the calculation

  :Returns:
    - step_size (float): Returns the value of the step size.
  """
  c = ( np.linalg.norm(u) * np.linalg.norm(gradient)**(-1) ) * 2 + 10
  merit = 0.5 * (np.linalg.norm(u))**2 + c * np.absolute(G)

  ntrial = 6
  """
  .. note::

     TODO: change fix value to a variable
  """

  Trial_step_size = np.array([0.5**np.arange(0,ntrial)])

  uT = np.reshape([u],(len(u),-1))
  dT = np.transpose(d)#np.reshape(d,(len(d),-1))
  zero = np.array([np.ones(ntrial)])
  zeroT = np.reshape(zero,(len(zero),-1))
  Trial_u = np.dot(uT,np.array([np.ones(ntrial)])) + np.dot(dT,Trial_step_size)
  Trial_x = np.zeros(Trial_u.shape)
  for j in range(ntrial):
    trial_x = u_to_x(Trial_u[:,j],stochastic_model)
    Trial_x[:,j] = np.transpose(trial_x)

  if analysis_options.getMultiProc() == 0:
    print 'Error: function not yet implemented'
  if analysis_options.getMultiProc() == 1:
    Trial_G, dummy = evaluateLimitState(Trial_x,stochastic_model,analysis_options,limit_state,'no')
    Merit_new = np.zeros(ntrial)

    for j in range(ntrial):
      merit_new = 0.5 * (np.linalg.norm(Trial_u[:,j]))**2 + c * np.absolute(Trial_G[0][j])
      Merit_new[j]=merit_new

    trial_step_size = Trial_step_size[0][0]
    merit_new = Merit_new[0]

    j = 0


    while merit_new > merit and j < ntrial:
      trial_step_size = Trial_step_size[0][j];
      merit_new = Merit_new[j];
      j += 1
      if j == ntrial and merit_new > merit:
        if analysis_options.printOutput():
          print 'The step size has been reduced by a factor of 1/',2**ntrial
  step_size = trial_step_size
  return step_size
