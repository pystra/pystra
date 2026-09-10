"""Base classes for reliability analysis.

:class:`AnalysisObject` is the common base for FORM, SORM, and Monte
Carlo analysis classes.  :class:`AnalysisOptions` holds all
user-configurable parameters for these analyses.
"""

import numpy as np
from ..model import StochasticModel, LimitState
from ..dependence.transformation import Transformation
from ..dependence.correlation import set_modified_correlation_matrix

__all__ = ["AnalysisObject", "AnalysisOptions"]


class AnalysisObject:
    """Base class for reliability analysis objects (FORM, SORM, MC).

    .. note::
        Subclasses should use ``self.N_HYPH`` for the width of console
        separator lines printed by ``show_results()``.

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

    N_HYPH = 58  # Width of console separator lines in show_results()

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
        selected = self.options.get_transform()
        self.transform = Transformation(
            transform_type=None if selected in ("nataf", "rosenblatt") else selected
        )

        self.results_valid = False

    def init_run(self):
        """Initialise the model's isoprobabilistic transformation.

        Uses the explicit copula or calibrates the legacy Gaussian Nataf
        correlation. Must be called at the start of every
        ``run()`` method in subclasses.
        """

        copula = self.model.get_copula()
        selected = self.options.get_transform()
        if copula is not None or selected in ("nataf", "rosenblatt"):
            from ..dependence.copula import GaussianCopula, StudentTCopula
            from ..dependence.joint import CopulaTransformation

            joint = self.model.get_joint_distribution()
            gaussian = isinstance(joint.copula, GaussianCopula) and not isinstance(
                joint.copula, StudentTCopula
            )
            method = selected
            if selected is None:
                method = "nataf" if gaussian else "rosenblatt"
            if selected in ("cholesky", "svd"):
                method = "nataf"
            self.transform = CopulaTransformation(
                joint,
                method=method,
                order=self.options.rosenblatt_order,
                factorization="svd" if selected == "svd" else "cholesky",
            )
            if self.transform.standard_space != "normal" and not getattr(
                self, "supports_spherical_space", False
            ):
                self.results_valid = False
                raise ValueError(
                    "This analysis requires independent normal space; select Rosenblatt for this copula"
                )
            if gaussian:
                self.model.set_modified_correlation(joint.copula.correlation)
            if self.options.get_print_output():
                print(
                    f"Using {type(joint.copula).__name__}, {method} transformation, {self.transform.standard_space} space"
                )
            return

        if self.options.rosenblatt_order is not None:
            raise ValueError(
                "Conditioning order requires the Rosenblatt transformation"
            )
        self.transform = Transformation(selected)
        if self.options.get_print_output():
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
        set_modified_correlation_matrix(self.model)

        # Compute the isoprobabilistic transform
        self.transform.compute(self.model.get_modified_correlation())


class AnalysisOptions:
    """Configuration for structural reliability analyses.

    All FORM, SORM, and Monte Carlo settings are collected here.
    Attributes can be set directly or via the legacy getter/setter
    methods.
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
        self.rosenblatt_order = None

    # getter
    def get_print_output(self):
        return self.print_output

    def get_flag_sens(self):
        return self.flag_sens

    def get_multi_proc(self):
        return self.multi_proc

    def get_block_size(self):
        return self.block_size

    def get_imax(self):
        return self.i_max

    def get_e1(self):
        return self.e1

    def get_e2(self):
        return self.e2

    def get_step_size(self):
        return self.step_size

    def get_diff_mode(self):
        return self.diff_mode

    def get_ffd_parameter(self):
        return self.ffdpara

    def get_samples(self):
        """
        Return the number of samples used in MCS
        """
        return self.samples

    def get_random_generator(self):
        return self.random_generator

    def get_simulation_point(self):
        return self.sim_point

    def get_simulation_stdv(self):
        return self.stdv_sim

    def get_simulation_cov(self):
        return self.target_cov

    def get_transform(self):
        return self.transform_type

    # setter
    def set_print_output(self, tof):
        self.print_output = tof

    def set_multi_proc(self, multi_proc):
        self.multi_proc = multi_proc

    def set_block_size(self, block_size):
        self.block_size = block_size

    def set_imax(self, i_max):
        self.i_max = i_max

    def set_e1(self, e1):
        self.e1 = e1

    def set_e2(self, e2):
        self.e2 = e2

    def set_step_size(self, step_size):
        self.step_size = step_size

    def set_diff_mode(self, diff_mode):
        self.diff_mode = diff_mode

    def set_ffd_parameter(self, ffdpara):
        self.ffdpara = ffdpara

    def set_bins(self, bins):
        self.bins = bins

    def set_samples(self, samples):
        """
        Set the number of samples used in MCS
        """
        self.samples = samples

    def set_transform(self, transform_type):
        """Select auto (None), cholesky/SVD Nataf, nataf, or rosenblatt."""
        if transform_type not in (None, "cholesky", "svd", "nataf", "rosenblatt"):
            raise ValueError("Unknown isoprobabilistic transformation")
        self.transform_type = transform_type

    def set_rosenblatt_order(self, order):
        """Set a permutation of stochastic variable indices for conditioning."""
        self.rosenblatt_order = None if order is None else tuple(order)
