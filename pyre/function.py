#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

def function(X1,X2,X3):
  g = 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2
  return g

