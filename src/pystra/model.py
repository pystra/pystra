# -*- coding: utf-8 -*-

import inspect

import numpy as np
from .distributions import Distribution, Constant
from collections import OrderedDict
from types import MappingProxyType
from .errors import ModelError, AnalysisError

__all__ = ["StochasticModel", "LimitState"]


class StochasticModel:
    """Stochastic model

    Random variables and constants are added with :meth:`add_variable`.
    ``model.variable(name)`` returns a random variable, and
    ``model.constants`` is a read-only mapping of constant names to values.
    """

    def __init__(self, joint_distribution=None):
        """
        Use ordered dictionary to make sure that the order corresponds to the
        correlation matrix
        """
        self.variables = OrderedDict()
        self._names = []
        self._marg = []
        self._correlation = None
        self._correlation_explicit = False
        self._Ro = None
        self._call_function = 0
        self._consts = {}
        self._copula = None
        if joint_distribution is not None:
            from .dependence.joint import JointDistribution

            if not isinstance(joint_distribution, JointDistribution):
                raise TypeError("Expected a JointDistribution")
            for marginal in joint_distribution.marginals:
                self.add_variable(marginal)
            self.set_copula(joint_distribution.copula)

    def add_variable(self, obj):
        """Add a random variable or constant to the model.

        Parameters
        ----------
        obj : Distribution or Constant
            The variable to add.  Distributions are treated as random
            variables; Constants are stored separately and passed as
            fixed values to the limit state function.

        Raises
        ------
        ModelError
            If *obj* is not a Distribution or Constant, or if a
            variable with the same name already exists. Add all random
            variables before explicitly setting correlation or a copula.
        """

        if not (isinstance(obj, Distribution) or isinstance(obj, Constant)):
            raise ModelError("Input is not a Distribution or Constant object")

        if obj.get_name() in self._names:
            raise ModelError(f'variable name "{obj.get_name()}" already exists')
        if isinstance(obj, Distribution) and self._copula is not None:
            raise ValueError("Add all random variables before setting the copula")
        if isinstance(obj, Distribution) and self._correlation_explicit:
            raise ModelError("Add all random variables before setting correlation")

        # append the variable name
        self._names.append(obj.get_name())

        if isinstance(obj, Distribution):
            # append marginal distribution
            self._marg.append(obj)
            # append the Distribution object to the variables (ordered) dictionary
            self.variables[obj.get_name()] = obj
            # update the default correlation matrix, in accordance with the number of variables
            self._correlation = np.eye(len(self._marg))
        elif isinstance(obj, Constant):
            self._consts[obj.get_name()] = obj.value

    # ---- Properties (preferred access) ----

    def __repr__(self):
        items = [repr(d) for d in self._marg] + [
            f"Constant({name!r}, value={value!r})"
            for name, value in self._consts.items()
        ]
        return f"StochasticModel([{', '.join(items)}])"

    @property
    def constants(self):
        """Read-only mapping of constant names to values."""
        return MappingProxyType(self._consts)

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
        return self.get_correlation()

    @correlation.setter
    def correlation(self, value):
        self.set_correlation(value)

    @property
    def copula(self):
        """Explicit dependence specification, or None for legacy Pearson input."""
        return self._copula

    @copula.setter
    def copula(self, value):
        self.set_copula(value)

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

    # ---- Transitional getter/setter methods pending the result/options redesign ----

    def get_variables(self):
        return self.variables

    def variable(self, name):
        """Return the random variable called *name*."""
        return self.variables[name]

    def get_names(self):
        return self._names

    def get_len_marginal_distributions(self):
        return len(self._marg)

    def get_marginal_distributions(self):
        return self._marg

    def set_marginal_distributions(self, marg):
        self._marg = marg

    def set_correlation(self, obj):
        """Set the physical correlation matrix, replacing any copula.

        Accepts a :class:`~pystra.CorrelationMatrix`, or an array that is
        valid as one, and raises :class:`~pystra.ModelError` otherwise.
        """
        from .dependence.correlation import CorrelationMatrix

        if not isinstance(obj, CorrelationMatrix):
            obj = CorrelationMatrix(obj)
        if obj.matrix.shape != (self.n_marg, self.n_marg):
            raise ModelError("Correlation dimensions must match the random variables")
        self._correlation = obj.matrix
        self._correlation_explicit = True
        self._copula = None
        self._Ro = None

    def get_correlation(self):
        if self._copula is not None:
            raise ValueError(
                "Physical Pearson correlation is not specified by an explicit copula; use get_copula()"
            )
        return self._correlation

    def set_copula(self, copula):
        """Replace legacy Pearson dependence with an explicit copula."""
        from .dependence.joint import JointDistribution

        JointDistribution(self._marg, copula)  # validate before changing state
        self._copula = copula
        self._correlation = None
        self._Ro = None

    def get_copula(self):
        return self._copula

    def get_joint_distribution(self):
        """Return marginals plus the explicit or calibrated Gaussian copula."""
        from .dependence.joint import JointDistribution
        from .dependence.copula import GaussianCopula
        from .dependence.correlation import compute_modified_correlation_matrix

        copula = self._copula
        if copula is None:
            copula = GaussianCopula(compute_modified_correlation_matrix(self))
        return JointDistribution(self._marg, copula)

    def set_modified_correlation(self, correlation):
        self._Ro = correlation

    def get_modified_correlation(self):
        return self._Ro

    def add_call_function(self, add):
        self._call_function += add

    def get_call_function(self):
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

    # Legacy getter/setter methods (expression is already a public attribute)

    def get_expression(self):
        return self.expression

    def set_expression(self, expression):
        self.expression = expression

    def evaluate_lsf(
        self,
        x,
        stochastic_model,
        *,
        differentiation="no",
        ffd_parameter=1000,
        block_size=1000,
        counter=None,
    ):
        """Evaluate the limit state function and (optionally) its gradient.

        Dispatches to the appropriate evaluation strategy based on the
        differentiation mode: no gradient (``"no"``), forward finite
        difference (``"ffd"``), or direct differentiation (``"ddm"``).
        The limit state keeps no evaluation state and ``x`` is not modified,
        so analyses sharing a limit state or model cannot interfere.

        Parameters
        ----------
        x : ndarray
            Evaluation points, shape ``(nrv, nx)`` where *nrv* is the
            number of random variables and *nx* the number of points.
        stochastic_model : StochasticModel
            The probabilistic model.
        differentiation : {"no", "ffd", "ddm"}, default "no"
            Values only, forward finite-difference gradients, or the gradient
            returned by the function itself (direct differentiation).
        ffd_parameter : float, default 1000
            Finite-difference step divisor: each variable is perturbed by its
            standard deviation divided by this value.
        block_size : int, default 1000
            Points passed to the limit-state function per call.
        counter : callable, optional
            Called with the number of limit-state function calls made, so
            an analysis can count its own evaluations.

        Returns
        -------
        G : ndarray
            Limit state function values, shape ``(1, nx)``.
        grad_G : ndarray
            Gradient matrix, shape ``(nrv, nx)``.  Zero when no
            gradient is computed.

        Raises
        ------
        ModelError
            Points or returned values/gradients have incompatible dimensions.
        AnalysisError
            Physical points, limit-state values or gradients are nonfinite,
            or the external evaluator raises. The evaluator exception is
            retained as the cause; no probability estimate is produced.
        """
        if differentiation not in ("no", "ffd", "ddm"):
            raise ValueError("differentiation must be 'no', 'ffd' or 'ddm'")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] != stochastic_model.n_marg:
            raise ModelError(
                "Evaluation points must have shape (n_variables, n_points)"
            )
        if not np.all(np.isfinite(x)):
            raise AnalysisError(
                "Limit-state evaluation requires finite physical points"
            )
        if differentiation == "no":
            G, grad_G, calls = self._values(x, stochastic_model, block_size)
        elif differentiation == "ddm":
            G, grad_G, calls = self._ddm(x, stochastic_model)
        else:
            G, grad_G, calls = self._ffd(x, stochastic_model, block_size, ffd_parameter)
        if not np.all(np.isfinite(grad_G)):
            raise AnalysisError(
                "Limit-state differentiation produced a nonfinite gradient"
            )
        stochastic_model.add_call_function(calls)
        if counter is not None:
            counter(calls)
        return G, grad_G

    def _values(self, x, model, block_size):
        """Limit-state values without gradients (used by simulation)."""
        nrv, nx = x.shape
        G = np.zeros((1, nx))
        grad_G = np.zeros((nrv, nx))
        k = 0
        while k < nx:
            block_size = np.min([block_size, nx - k])
            indx = list(range(k, k + block_size))
            blockG, _ = self._call(x[:, indx], model)
            G[:, indx] = blockG
            k += block_size
        return G, grad_G, nx

    def _ffd(self, x, model, block_size, ffdpara):
        """Limit-state values and forward finite-difference gradients."""
        nrv, nx = x.shape
        grad_G = np.zeros((nrv, nx))
        allx = np.repeat(x, 1 + nrv, axis=1)
        allh = np.zeros(nrv)

        marg = model.get_marginal_distributions()

        for j in range(nrv):
            allh[j] = marg[j].std / ffdpara
            indx = list(range(j + 1, 1 + (1 + j + (nx - 1) * (1 + nrv)), (1 + nrv)))
            allx[j, indx] = x[j] + allh[j] * np.ones(nx)

        allG = np.zeros(nx * (1 + nrv))

        k = 0
        while k < (nx * (1 + nrv)):
            block_size = np.min([block_size, nx * (1 + nrv) - k])
            indx = list(range(k, k + block_size))
            blockG, _ = self._call(allx[:, indx], model)
            allG[indx] = blockG.squeeze()
            k += block_size

        indx = list(range(0, (1 + (nx - 1) * (1 + nrv)), (1 + nrv)))
        G = allG[indx]

        for j in range(nrv):
            indx = list(range(j + 1, 1 + (1 + j + (nx - 1) * (1 + nrv)), (1 + nrv)))
            grad_G[j, :] = (allG[indx] - G) / allh[j]

        return G, grad_G, nx * (1 + nrv)

    def _ddm(self, x, model):
        """Limit-state values with the user-supplied gradient (direct differentiation)."""
        nrv, nx = x.shape
        G = np.zeros((1, nx))
        grad_G = np.zeros((nrv, nx))
        for k in range(nx):
            values, gradient = self._call(x[:, k : k + 1], model, ddm=True)
            G[:, k] = values
            grad_G[:, k] = gradient.reshape(nrv)
        return G, grad_G, nx

    def _call(self, x, model, ddm=False):
        """Call the user-defined limit state function.

        Builds a keyword-argument dictionary mapping variable names to their
        rows of ``x`` (and constants to matching vectors), then calls
        ``self.expression(**kwargs)``. With ``ddm`` the expression must return
        both the function value and a gradient vector.
        """
        _, nc = np.shape(x)
        variables = model.get_variables()
        constants = model.constants

        inpdict = dict()
        for i, var in enumerate(variables):
            inpdict[var] = x[i]
        for c, val in constants.items():
            inpdict[c] = val * np.ones(nc)
        # Binding errors describe the model, whereas exceptions from inside
        # a correctly bound evaluator describe a failed analysis.
        signature = self._signature()
        if signature is not None:
            try:
                signature.bind(**inpdict)
            except TypeError as error:
                raise ModelError(
                    f"Limit-state signature does not match model: {error}"
                ) from error
        context = f"{nc} point(s), first point {x[:, 0].tolist()}"
        try:
            Gvals = self.expression(**inpdict)
        except Exception as error:
            raise AnalysisError(
                f"Limit-state evaluation failed at {context}: {error}"
            ) from error
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

        G = np.asarray(G, dtype=float)
        if G.ndim == 0:
            G = np.full(nc, G.item())
        elif G.shape == (1, nc):
            G = G[0]
        elif G.shape != (nc,):
            raise ModelError(
                f"Limit-state values must have shape ({nc},), got {G.shape}"
            )
        if not np.all(np.isfinite(G)):
            raise AnalysisError(f"Nonfinite limit-state values at {context}")
        if ddm:
            gradient = np.asarray(gradient, dtype=float)
            if gradient.shape not in ((len(variables),), (len(variables), 1)):
                raise ModelError(
                    "DDM gradient must contain one derivative per random variable"
                )
            if not np.all(np.isfinite(gradient)):
                raise AnalysisError(f"Nonfinite limit-state gradient at {context}")
        return G, gradient

    def _signature(self):
        """Signature of the expression, kept until the expression changes."""
        if not callable(self.expression):
            raise ModelError("Limit-state expression must be callable")
        if getattr(self, "_signature_of", None) is not self.expression:
            try:
                signature = inspect.signature(self.expression)
            except (TypeError, ValueError):
                signature = None  # Some extension callables expose no signature.
            self._signature_of, self._expression_signature = self.expression, signature
        return self._expression_signature

    def __getstate__(self):
        # The cached signature is derived, and its defaults need not pickle
        state = self.__dict__.copy()
        state.pop("_signature_of", None)
        state.pop("_expression_signature", None)
        return state
