#!/usr/bin/python -tt
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
        self.Lo = None
        self.iLo = None
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

    def setLowerTriangularMatrix(self, matrix):
        self.Lo = matrix

    def getLowerTriangularMatrix(self):
        return self.Lo

    def setInvLowerTriangularMatrix(self, matrix):
        self.iLo = matrix

    def getInvLowerTriangularMatrix(self):
        return self.iLo

    def addCallFunction(self, add):
        self.call_function += add

    def getCallFunction(self):
        return self.call_function


class AnalysisOptions:
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

        self.Ro_method = 1
        """Method for computation of the modified Nataf correlation matrix

    :Methods:
      - 0: use of approximations from ADK's paper (no longer supported)\n
      - 1: exact, solved numerically
    """

        self.flag_sens = True
        """ Flag for computation of sensitivities

    w.r.t. means, standard deviations, parameters and correlation coefficients

    :Flag:
      - 1: all sensitivities assessed,\n
      - 0: no sensitivities assessment
    """

        self.print_output = False
        """Print output to the console during calculation

    :Values:
      - True: prints output to the console (useful, e.g. spyder),\n
      - False: does not print out (e.g. jupyter notebook)
    """

        self.multi_proc = 1
        """ Amount of g-calls

    1: block_size g-calls sent simultaneously
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
        self.diff_mode = "ffd"
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
        # 100 for FE-based limit-state functions

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
        self.sim_point = "origin"
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
    def getPrintOutput(self):
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

    def getDiffMode(self):
        return self.diff_mode

    def getffdpara(self):
        return self.ffdpara

    def getBins(self):
        if self.bins is not None:
            return self.bins
        else:
            bins = np.ceil(4 * np.sqrt(np.sqrt(self.samples)))
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
    def setPrintOutput(self, tof):
        self.print_output = tof

    def setMultiProc(self, multi_proc):
        self.multi_proc = multi_proc

    def setBlockSize(self, block_size):
        self.block_size = block_size

    def setImax(self, i_max):
        self.i_max = i_max

    def setE1(self, e1):
        self.e1 = e1

    def setE2(self, e2):
        self.e2 = e2

    def setStepSize(self, step_size):
        self.step_size = step_size

    def setDiffMode(self, diff_mode):
        self.diff_mode = diff_mode

    def setffdpara(self, ffdpara):
        self.ffdpara = ffdpara

    def setBins(self, bins):
        self.bins = bins

    def setSamples(self, samples):
        self.samples = samples


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


class AnalysisObject:
    """
    A base class for objects that perform a probability of failure estimation
    """

    def __init__(self, analysis_options=None, limit_state=None, stochastic_model=None):
        # The stochastic model
        if stochastic_model is None:
            self.model = StochasticModel()
        else:
            self.model = stochastic_model

        # Options for the calculation
        if analysis_options is None:
            self.options = AnalysisOptions()
        else:
            self.options = analysis_options

        # The limit state function
        if limit_state is None:
            self.limitstate = LimitState()
        else:
            self.limitstate = limit_state

        self.results_valid = False
