# -*- coding: utf-8 -*-

import numpy as np
from .distributions import Distribution, Constant
from collections import OrderedDict


class StochasticModel:
    """Stochastic model"""

    def __init__(self):
        """
        Use ordered dictionary to make sure that the order corresponds to the
        correlation matrix
        """
        self.variables = OrderedDict()
        self.names = []
        self.marg = []
        self.correlation = None
        self.Ro = None
        self.call_function = 0
        self.consts = {}

    def addVariable(self, obj):
        """
        add stochastic variable
        """

        if not (isinstance(obj, Distribution) or isinstance(obj, Constant)):
            raise Exception("Input is not a Distribution or Constant object")

        if obj.getName() in self.names:
            raise Exception(f'variable name "{obj.getName()}" already exists')

        # append the variable name
        self.names.append(obj.getName())

        if isinstance(obj, Distribution):
            # append marginal distribution
            self.marg.append(obj)
            # append the Distribution object to the variables (ordered) dictionary
            self.variables[obj.getName()] = obj
            # update the default correlation matrix, in accordance with the number of variables
            self.correlation = np.eye(len(self.marg))
        elif isinstance(obj, Constant):
            self.consts[obj.getName()] = obj.getValue()

    def getConstants(self):
        return self.consts

    def getVariables(self):
        return self.variables

    def getVariable(self, name):
        return self.variables[name]

    def getNames(self):
        return self.names

    def getLenMarginalDistributions(self):
        return len(self.marg)

    def getMarginalDistributions(self):
        return self.marg

    def setMarginalDistributions(self, marg):
        self.marg = marg

    def setCorrelation(self, obj):
        self.correlation = np.array(obj.getMatrix())

    def getCorrelation(self):
        return self.correlation

    def setModifiedCorrelation(self, correlation):
        self.Ro = correlation

    def getModifiedCorrelation(self):
        return self.Ro

    def addCallFunction(self, add):
        self.call_function += add

    def getCallFunction(self):
        return self.call_function


class LimitState:
    r"""
    The Limit State function definition class.

    The limit state function can be defined in two main ways:

    1. Numerical differentiation (FFD): the limit state function need only return
    its value at a set of evaluation points, X. In this form, the function can be
    either:

        (a) A python lambda object;
        (b) A python function object.

    2. Using the Direct Differentiation Method (DDM): the limit state function
    is a python function object return both its value and gradient vector at each
    of the evaluation points.

    Note in both cases that each parameter (i.e. function argument) may be passed
    as a vector, depending on the algorithm being called.

    Where a function returns a gradient vector, it is only utilized when DDM is
    specified.
    """

    def __init__(self, expression=None):
        self.expression = expression
        """Expression of the limit-state function"""

        self.model = None
        self.options = None
        self.x = None
        self.nx = 0
        self.nrv = 0

    def getExpression(self):
        return self.expression

    def setExpression(self, expression):
        self.expression = expression

    def evaluate_lsf(self, x, stochastic_model, analysis_options, diff_mode=None):
        """Evaluate the limit state"""

        self.model = stochastic_model
        self.options = analysis_options

        self.x = x

        if diff_mode == None:
            diff_mode = analysis_options.getDiffMode()
        else:
            diff_mode = "no"

        if analysis_options.getMultiProc() == 0:
            raise NotImplementedError("getMultiProc")
        else:
            # No differentiation for MCS
            if diff_mode == "no":
                G, grad_G = self.evaluate_nogradient(x)
            elif diff_mode == "ddm":
                G, grad_G = self.evaluate_ddm(x)
            else:
                G, grad_G = self.evaluate_ffd(x)
        return G, grad_G

    def evaluate_nogradient(self, x):
        nrv, nx = x.shape
        G = np.zeros((1, nx))
        grad_G = np.zeros((nrv, nx))
        block_size = self.options.getBlockSize()
        if nx > 1:
            k = 0
            while k < nx:
                block_size = np.min([block_size, nx - k])
                indx = list(range(k, k + block_size))
                blockx = x[:, indx]

                blockG, _ = self.compute_lsf(blockx)

                G[:, indx] = blockG
                # grad_g[indx] = blockdummy
                k += block_size

            self.model.addCallFunction(nx)

        return G, grad_G

    def evaluate_ffd(self, x):
        nrv, nx = x.shape
        G = np.zeros((1, nx))
        grad_G = np.zeros((nrv, nx))
        block_size = self.options.getBlockSize()

        ffdpara = self.options.getffdpara()
        allx = np.zeros((nrv, nx * (1 + nrv)))
        allx[:] = x
        allh = np.zeros(nrv)

        marg = self.model.getMarginalDistributions()

        x0 = x
        for j in range(nrv):
            x = x0
            allh[j] = marg[j].stdv / ffdpara
            x[j] = x[j] + allh[j] * np.ones(nx)
            indx = list(range(j + 1, 1 + (1 + j + (nx - 1) * (1 + nrv)), (1 + nrv)))
            allx[j, indx] = x[j]

        allG = np.zeros(nx * (1 + nrv))

        k = 0
        while k < (nx * (1 + nrv)):
            block_size = np.min([block_size, nx * (1 + nrv) - k])
            indx = list(range(k, k + block_size))
            blockx = allx[:, indx]

            blockG, _ = self.compute_lsf(blockx)

            allG[indx] = blockG.squeeze()
            k += block_size

        indx = list(range(0, (1 + (nx - 1) * (1 + nrv)), (1 + nrv)))
        G = allG[indx]

        for j in range(nrv):
            indx = list(range(j + 1, 1 + (1 + j + (nx - 1) * (1 + nrv)), (1 + nrv)))
            grad_G[j, :] = (allG[indx] - G) / allh[j]

        self.model.addCallFunction(nx * (1 + nrv))

        return G, grad_G

    def evaluate_ddm(self, x):
        nrv, nx = x.shape
        G = np.zeros((1, nx))
        grad_G = np.zeros((nrv, nx))
        for k in range(nx):
            G[k], grad_G[:, k : k + 1] = self.compute_lsf(x[:, k : k + 1], ddm=True)
        self.model.addCallFunction(nx)

        return G, grad_G

    def compute_lsf(self, x, ddm=False):
        """Compute the limit state function"""
        _, nc = np.shape(x)
        variables = self.model.getVariables()
        constants = self.model.getConstants()

        inpdict = dict()
        for i, var in enumerate(variables):
            inpdict[var] = x[i]
        for c, val in constants.items():
            inpdict[c] = val * np.ones(nc)
        Gvals = self.expression(**inpdict)
        try:
            if ddm:
                G, gradient = Gvals
            else:
                if isinstance(Gvals, tuple):
                    G = Gvals[0]
                else:
                    G = Gvals
                gradient = 0
        except TypeError:
            raise TypeError(
                "Limit state function return must match differentiation mode"
            )

        return G, gradient
