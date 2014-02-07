#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import os, sys
from types import FunctionType, StringType

sys.path.append (os.path.join(os.getcwd(), "functions") )
from functions.function import *

def evaluateLimitState(x,stochastic_model,analysis_options,limit_state,modus=None):
  """Evaluate the limit state"""

  global nfun
  names = stochastic_model.getNames()
  expression = limit_state.getExpression()

  nx = x.shape[1]
  nrv = x.shape[0]

  if modus == None:
    modus =  analysis_options.getDifferentationModus()
  else:
    modus = 'no'

  if analysis_options.getMultiProc() == 0:
    print 'Error: function not yet implemented'
  if analysis_options.getMultiProc() == 1:
    block_size = analysis_options.getBlockSize()
    if modus == 'no':
      if nx > 1:
        G = np.zeros((1,nx))
        dummy = np.zeros(nx)
        k = 0
        while k < nx:
          block_size = np.min([block_size,nx-k])
          indx = range(k,k+block_size)
          blockx = x[:,indx]

          if limit_state.getEvaluator() == 'basic':
            blockG,blockdummy = computeLimitStateFunction(blockx,names,expression)

          G[:,indx] = blockG
          dummy[indx] = blockdummy 
          k += block_size
        grad_g = dummy
        stochastic_model.addCallFunction(nx)
    elif modus == 'ddm':
      print 'Error: ddm function not yet implemented'
    elif modus == 'ffd':
      ffdpara = analysis_options.getffdpara()
      allx = np.zeros((nrv,nx*(1+nrv)))
      # indx = range(0,(1+(nx-1)*(1+nrv)),(1+nrv))
      # allx[:,indx] = x
      allx[:] = x
      allh = np.zeros(nrv)

      marg = stochastic_model.getMarginalDistributions()

      original_x = x

      for j in range(nrv):
        x = original_x
        # TODO marg
        allh[j] = marg[j][2]*ffdpara**(-1)
        x[j]= x[j] + allh[j]*np.ones(nx)
        indx = range(j+1,1+(1+j+(nx-1)*(1+nrv)),(1+nrv))
        allx[j,indx] = x[j]

      allG = np.zeros(nx*(1+nrv))

      k = 0
      while k < (nx*(1+nrv)):
        block_size = np.min([block_size,nx*(1+nrv)-k])
        indx = range(k,k+block_size)
        blockx = allx[:,indx]

        if limit_state.getEvaluator() == 'basic':
          blockG ,dummy = computeLimitStateFunction(blockx,names,expression)
            
        allG[indx] = blockG.squeeze()
        k += block_size

      indx = range(0,(1+(nx-1)*(1+nrv)),(1+nrv))
      G = allG[indx]
      grad_g = np.zeros((nrv,nx))

      for j in range(nrv):
        indx = range(j+1,1+(1+j+(nx-1)*(1+nrv)),(1+nrv))
        grad_g[j,:] = (allG[indx] - G) * (allh[j])**(-1)

      stochastic_model.addCallFunction(nx*(1+nrv))

  return G, grad_g


def computeLimitStateFunction(x,variable_names,expression):
  """Compute the limit state function"""
  nrv = np.shape(x)[0]

  if type(expression) == StringType:
    # string expression, for backward compatibility
    for i in range(nrv):
      globals()[variable_names[i]] = x[i:i+1]
    G = eval(expression)[0]
  elif type(expression) == FunctionType:
    # function expression, recommended to use
    inpdict = dict()
    for i in range(nrv):
      inpdict[variable_names[i]] = x[i:i+1]
    G = expression(**inpdict)
  gradient = 0
  return G,gradient




