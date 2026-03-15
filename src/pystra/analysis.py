"""Base classes for reliability analysis.

:class:`AnalysisObject` is the common base for FORM, SORM, and Monte
Carlo analysis classes.  :class:`AnalysisOptions` holds all
user-configurable parameters for these analyses.
"""

import numpy as np
from .model import StochasticModel, LimitState
from .transformation import Transformation
from .correlation import setModifiedCorrelationMatrix


class AnalysisObject:
    """Base class for reliability analysis objects (FORM, SORM, MC).

    .. note::
        Subclasses should use ``self.N_HYPH`` for the width of console
        separator lines printed by ``showResults()``.

    Handles the common set-up shared by all analysis types: storing the
    stochastic model, limit state, and analysis options, and providing
    the ``init_run`` method that computes the Nataf correlation and
    isoprobabilistic transformation before the analysis-specific
    iteration begins.

    Parameters
    ----------
    stochastic_model : StochasticModel, optional
        The probabilistic model.
    limit_state : LimitState, optional
        The limit state function.
    analysis_options : AnalysisOptions, optional
        Algorithm settings.

    Attributes
    ----------
    model : StochasticModel
    limitstate : LimitState
    options : AnalysisOptions
    transform : Transformation
    results_valid : bool
        ``True`` after a successful ``run()``.
    """

    N_HYPH = 58  # Width of console separator lines in showResults()

    def __init__(self, stochastic_model=None, limit_state=None, analysis_options=None):
        # The stochastic model
        if stochastic_model is None:
            self.model = StochasticModel()
        else:
            self.model = stochastic_model

        # The limit state function
        if limit_state is None:
            self.limitstate = LimitState()
        else:
            self.limitstate = limit_state

        # Options for the calculation
        if analysis_options is None:
            self.options = AnalysisOptions()
        else:
            self.options = analysis_options

        # Create transformation based on user settings in AnalysisOptions
        self.transform = Transformation(transform_type=self.options.getTransform())

        self.results_valid = False

    def init_run(self):
        """Initialise the Nataf transformation before the analysis loop.

        Computes the modified (Nataf) correlation matrix and its
        factorisation.  Must be called at the start of every
        ``run()`` method in subclasses.
        """

        if self.options.getPrintOutput():
            print("==================================================")
            print("")
            print("           RUNNING RELIABILITY ANALYSIS")
            print("")
            print("==================================================")
            print("")
            print(" Computation of modified correlation matrix R0")
            print(" Takes some time if sensitivities are to be computed")
            print(" with gamma (3), beta (7) or chi-square (8)")
            print(" distributions.")
            print(" Please wait... (Ctrl+C breaks)")
            print("")

        # Computation of modified correlation matrix R0
        setModifiedCorrelationMatrix(self.model)

        # Compute the isoprobabilistic transform
        self.transform.compute(self.model.getModifiedCorrelation())


class AnalysisOptions:
    """Configuration for structural reliability analyses.

    All FORM, SORM, and Monte Carlo settings are collected here.
    Attributes can be set directly or via the legacy getter/setter
    methods.

    Attributes
    ----------
    print_output : bool
        Print progress to the console (default ``False``).
    diff_mode : {"ffd", "ddm"}
        Gradient computation method — forward finite difference or
        direct differentiation (default ``"ffd"``).
    ffdpara : int
        FFD perturbation divisor; perturbation = ``stdv / ffdpara``
        (default 1000).
    i_max : int
        Maximum FORM iterations (default 100).
    e1 : float
        Convergence tolerance on the limit state value (default 0.001).
    e2 : float
        Convergence tolerance on the gradient direction (default 0.001).
    step_size : float
        FORM step size (0 = Armijo rule, default 0).
    samples : int
        Number of Monte Carlo samples (default 100 000).
    target_cov : float
        Target coefficient of variation for MC failure probability
        (default 0.05).
    transform_type : str or None
        Isoprobabilistic transform type (``"cholesky"`` or ``"svd"``);
        ``None`` uses the default (Cholesky).
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

        self.transform_type = None

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

    def getSamples(self):
        """
        Return the number of samples used in MCS
        """
        return self.samples

    def getRandomGenerator(self):
        return self.random_generator

    def getSimulationPoint(self):
        return self.sim_point

    def getSimulationStdv(self):
        return self.stdv_sim

    def getSimulationCov(self):
        return self.target_cov

    def getTransform(self):
        return self.transform_type

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
        """
        Set the number of samples used in MCS
        """
        self.samples = samples

    def setTransform(self, transform_type):
        self.transform_type = transform_type
