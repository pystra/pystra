# -*- coding: utf-8 -*-

import numpy as np
from .distributions import Distribution, Constant
from collections import OrderedDict


class StochasticModel:
    """Stochastic model

    Attributes can be accessed via properties or the legacy getter methods::

        model.constants          # preferred
        model.getConstants()     # legacy, equivalent
    """

    def __init__(self):
        """
        Use ordered dictionary to make sure that the order corresponds to the
        correlation matrix
        """
        self.variables = OrderedDict()
        self._names = []
        self._marg = []
        self._correlation = None
        self._Ro = None
        self._call_function = 0
        self._consts = {}

    def addVariable(self, obj):
        """Add a random variable or constant to the model.

        Parameters
        ----------
        obj : Distribution or Constant
            The variable to add.  Distributions are treated as random
            variables; Constants are stored separately and passed as
            fixed values to the limit state function.

        Raises
        ------
        Exception
            If *obj* is not a Distribution or Constant, or if a
            variable with the same name already exists.
        """

        if not (isinstance(obj, Distribution) or isinstance(obj, Constant)):
            raise Exception("Input is not a Distribution or Constant object")

        if obj.getName() in self._names:
            raise Exception(f'variable name "{obj.getName()}" already exists')

        # append the variable name
        self._names.append(obj.getName())

        if isinstance(obj, Distribution):
            # append marginal distribution
            self._marg.append(obj)
            # append the Distribution object to the variables (ordered) dictionary
            self.variables[obj.getName()] = obj
            # update the default correlation matrix, in accordance with the number of variables
            self._correlation = np.eye(len(self._marg))
        elif isinstance(obj, Constant):
            self._consts[obj.getName()] = obj.getValue()

    # ---- Properties (preferred access) ----

    @property
    def constants(self):
        """Dictionary of constant name → value pairs."""
        return self._consts

    @property
    def names(self):
        """List of all variable and constant names, in insertion order."""
        return self._names

    @property
    def n_marg(self):
        """Number of marginal (stochastic) distributions."""
        return len(self._marg)

    @property
    def marginal_distributions(self):
        """List of marginal Distribution objects."""
        return self._marg

    @property
    def correlation(self):
        """Correlation matrix (n × n numpy array)."""
        return self._correlation

    @correlation.setter
    def correlation(self, value):
        self._correlation = value

    @property
    def modified_correlation(self):
        """Modified (Nataf) correlation matrix Ro."""
        return self._Ro

    @modified_correlation.setter
    def modified_correlation(self, value):
        self._Ro = value

    @property
    def call_function(self):
        """Cumulative number of limit-state function evaluations."""
        return self._call_function

    @call_function.setter
    def call_function(self, value):
        self._call_function = value

    # ---- Legacy getter/setter methods (kept for backward compatibility) ----

    def getConstants(self):
        return self._consts

    def getVariables(self):
        return self.variables

    def getVariable(self, name):
        return self.variables[name]

    def getNames(self):
        return self._names

    def getLenMarginalDistributions(self):
        return len(self._marg)

    def getMarginalDistributions(self):
        return self._marg

    def setMarginalDistributions(self, marg):
        self._marg = marg

    def setCorrelation(self, obj):
        self._correlation = np.array(obj.getMatrix())

    def getCorrelation(self):
        return self._correlation

    def setModifiedCorrelation(self, correlation):
        self._Ro = correlation

    def getModifiedCorrelation(self):
        return self._Ro

    def addCallFunction(self, add):
        self._call_function += add

    def getCallFunction(self):
        return self._call_function


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

    **Argument matching**: the function is called as ``expression(**kwargs)``
    where each keyword argument is a variable name from the
    :class:`StochasticModel`.  Arguments may therefore be declared explicitly
    (``def lsf(X1, X2, X3): ...``) *or* collected with ``**kwargs`` for a
    dimension-agnostic definition::

        def lsf(**kwargs):
            return sum(v**2 for v in kwargs.values())
    """

    def __init__(self, expression=None):
        self.expression = expression
        """Expression of the limit-state function"""

        self.model = None
        self.options = None
        self.x = None
        self.nx = 0
        self.nrv = 0

    # Legacy getter/setter methods (expression is already a public attribute)

    def getExpression(self):
        return self.expression

    def setExpression(self, expression):
        self.expression = expression

    def evaluate_lsf(self, x, stochastic_model, analysis_options, diff_mode=None):
        """Evaluate the limit state function and (optionally) its gradient.

        Dispatches to the appropriate evaluation strategy based on the
        differentiation mode: no gradient (``"no"``), forward finite
        difference (``"ffd"``), or direct differentiation (``"ddm"``).

        Parameters
        ----------
        x : ndarray
            Evaluation points, shape ``(nrv, nx)`` where *nrv* is the
            number of random variables and *nx* the number of points.
        stochastic_model : StochasticModel
            The probabilistic model.
        analysis_options : AnalysisOptions
            Algorithm settings (differentiation mode, block size, etc.).
        diff_mode : str or None, optional
            Override the differentiation mode.  If a string is passed
            (any value), gradient computation is suppressed
            (``"no"``).

        Returns
        -------
        G : ndarray
            Limit state function values, shape ``(1, nx)``.
        grad_G : ndarray
            Gradient matrix, shape ``(nrv, nx)``.  Zero when no
            gradient is computed.
        """

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
        """Evaluate the LSF without computing gradients (used for MCS)."""
        nrv, nx = x.shape
        G = np.zeros((1, nx))
        grad_G = np.zeros((nrv, nx))
        block_size = self.options.getBlockSize()
        k = 0
        while k < nx:
            block_size = np.min([block_size, nx - k])
            indx = list(range(k, k + block_size))
            blockx = x[:, indx]

            blockG, _ = self.compute_lsf(blockx)

            G[:, indx] = blockG
            k += block_size

        self.model.addCallFunction(nx)

        return G, grad_G

    def evaluate_ffd(self, x):
        """Evaluate the LSF and approximate the gradient by forward finite difference."""
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
        """Evaluate the LSF using direct differentiation (user-supplied gradient)."""
        nrv, nx = x.shape
        G = np.zeros((1, nx))
        grad_G = np.zeros((nrv, nx))
        for k in range(nx):
            G[k], grad_G[:, k : k + 1] = self.compute_lsf(x[:, k : k + 1], ddm=True)
        self.model.addCallFunction(nx)

        return G, grad_G

    def compute_lsf(self, x, ddm=False):
        """Call the user-defined limit state function.

        Builds a keyword-argument dictionary mapping variable names
        to their column vectors in ``x``, then calls
        ``self.expression(**kwargs)``.

        Parameters
        ----------
        x : ndarray
            Evaluation points, shape ``(nrv, nc)``.
        ddm : bool, optional
            If ``True``, expects the expression to return both the
            function value and a gradient vector.

        Returns
        -------
        G : ndarray
            Function value(s).
        gradient : ndarray or int
            Gradient vector (if *ddm*) or ``0``.
        """
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
