#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm as normal
from .analysis import AnalysisObject
from .correlation import setModifiedCorrelationMatrix


class Form(AnalysisObject):
    r"""First Order Reliability Method (FORM)

    Let :math:`{\\bf Z}` be a set of uncorrelated and standardized normally
    distributed random variables :math:`( Z_1 ,\dots, Z_n )` in the normalized
    z-space, corresponding to any set of random variables :math:`{\\bf X} = (
    X_1 , \dots , X_n )` in the physical x-space, then the limit state surface
    in x-space is also mapped on the corresponding limit state surface in
    z-space.

    The reliability index :math:`\\beta` is the minimum distance from the
    z-origin to the failure surface. This distance :math:`\\beta` can directly
    be mapped to a probability of failure

    .. math::

              p_f \\approx p_{f1} = \Phi(-\\beta)

    this corresponds to a linearization of the failure surface. The
    linearization point is the design point :math:`{\\bf z}^*`. This procedure
    is called First Order Reliability Method (FORM) and :math:`\beta` is the
    First Order Reliability Index. [Madsen2006]_

    :Attributes:
      - stochastic_model (StochasticModel): Information about the model
      - limit_state (LimitState): Information about the limit state
      - analysis_option (AnalysisOption): Option for the structural analysis
    """

    def __init__(self, stochastic_model=None, limit_state=None, analysis_options=None):

        super().__init__(
            stochastic_model=stochastic_model,
            limit_state=limit_state,
            analysis_options=analysis_options,
        )

        self.i = None
        self.u = None
        self.x = None
        self.J = None
        self.G = None
        self.Go = None
        self.gradient = None
        self.alpha = None
        self.gamma = None
        self.d = None
        self.step = None
        self.beta = None
        self.Pf = None

    def run(self):
        """
        Executes the FORM analysis
        """
        self.results_valid = True

        self.init_run()

        # Compute starting point for the algorithm
        self.computeStartingPoint()

        # Iterations
        # Set parameters for the iterative loop
        # Initialize counter
        i = 1
        # Convergence is achieved when convergence is set to True
        convergence = False

        # loope
        while not convergence:
            if self.options.getPrintOutput():
                print(".......................................")
                print("Now carrying out iteration number:", i)

            # Compute Transformation from u to x space
            self.computeTransformation()

            # Compute the Jacobian
            self.computeJacobian()

            # Evaluate limit-state function and its gradient
            self.computeLimitState()

            # Set scale parameter Go and inform about struct. resp.
            if i == 1:
                self.Go = self.G
                if self.options.getPrintOutput():
                    print("Value of limit-state function in the first step:", self.G)

            # Compute alpha vector
            self.computeAlpha()

            # Compute gamma vector
            self.computeGamma()

            # Check convergence
            e1 = np.absolute(self.G * self.Go ** (-1))[0]
            e2 = np.linalg.norm(self.u - self.alpha.dot(self.u).dot(self.alpha))
            condition1 = e1 < self.options.getE1()
            condition2 = e2 < self.options.getE2()
            condition3 = i == self.options.getImax()
            if self.options.getPrintOutput():
                print(f"e1 = {e1:1.6e} , e2 = {e2:1.6e}")

            if condition1 and condition2 or condition3:
                self.i = i
                convergence = True

            # space for some recording stuff

            # Take a step if convergence is not achieved
            if not convergence:
                # Determine search direction
                self.computeSearchDirection()

                # Determine step size
                self.getStepSize()

                # Determine new trial point
                u_new = self.u + self.step * self.d

                # Prepare for a new round in the loop
                self.u = u_new[0]  # np.transpose(u_new)
                i += 1

        # Compute beta value
        self.computeBeta()

        # Compute failure probability
        self.computeFailureProbability()

        # Show Results
        if self.options.getPrintOutput():
            self.showResults()

    def computeStartingPoint(self):
        """Compute starting point for the algorithm"""
        x = np.array([])
        marg = self.model.getMarginalDistributions()
        for i in range(len(marg)):
            x = np.append(x, marg[i].getStartPoint())
        self.u = self.transform.x_to_u(x, marg)

    def computeTransformation(self):
        """Compute transformation from u to x space"""
        self.x = np.transpose(
            [self.transform.u_to_x(self.u, self.model.getMarginalDistributions())]
        )

    def computeJacobian(self):
        """Compute the Jacobian"""
        J_u_x = self.transform.jacobian(
            self.u, self.x, self.model.getMarginalDistributions()
        )
        J_x_u = np.linalg.inv(J_u_x)
        self.J = J_x_u

    def computeLimitState(self):
        """Evaluate limit-state function and its gradient"""
        G, gradient = self.limitstate.evaluate_lsf(self.x, self.model, self.options)
        self.G = G
        self.gradient = np.dot(np.transpose(gradient), self.J)

    def computeAlpha(self):
        """Compute alpha vector"""
        self.alpha = -self.gradient * np.linalg.norm(self.gradient) ** (-1)

    def computeGamma(self):
        """Compute gamma vector"""
        self.gamma = np.diag(np.diag(np.sqrt(np.dot(self.J, np.transpose(self.J)))))
        # Importance vector gamma
        # matmult = np.dot(np.dot(self.alpha, self.J), self.gamma)
        # importance_vector_gamma = matmult / np.linalg.norm(matmult)

    def computeSearchDirection(self):
        """Determine search direction"""
        self.d = (
            self.G * np.linalg.norm(self.gradient) ** (-1) + self.alpha.dot(self.u)
        ) * self.alpha - self.u

    def getStepSize(self):
        """Determine step size"""
        if self.options.getStepSize() == 0:
            self.step = self.computeStepSize(
                self.G,
                self.gradient,
                self.u,
                self.d,
            )
        else:
            self.step = self.options.getStepSize()

    def computeStepSize(self, G, gradient, u, d):
        """Calculate the step size for the calculation

        :Returns:
            - step_size (float): Returns the value of the step size.
        """
        c = (np.linalg.norm(u) * np.linalg.norm(gradient) ** (-1)) * 2 + 10
        merit = 0.5 * (np.linalg.norm(u)) ** 2 + c * np.absolute(G)

        ntrial = 6
        """
        .. note::

             TODO: change fix value to a variable
        """

        Trial_step_size = np.array([0.5 ** np.arange(0, ntrial)])

        uT = np.reshape([u], (len(u), -1))
        dT = np.transpose(d)  # np.reshape(d,(len(d),-1))
        # zero = np.array([np.ones(ntrial)])
        # zeroT = np.reshape(zero, (len(zero), -1))
        Trial_u = np.dot(uT, np.array([np.ones(ntrial)])) + np.dot(dT, Trial_step_size)
        Trial_x = np.zeros(Trial_u.shape)
        for j in range(ntrial):
            trial_x = self.transform.u_to_x(
                Trial_u[:, j], self.model.getMarginalDistributions()
            )
            Trial_x[:, j] = np.transpose(trial_x)

        if self.options.getMultiProc() == 0:
            print("Error: function not yet implemented")
        if self.options.getMultiProc() == 1:
            Trial_G, _ = self.limitstate.evaluate_lsf(
                Trial_x, self.model, self.options, "no"
            )
            Merit_new = np.zeros(ntrial)

            for j in range(ntrial):
                merit_new = 0.5 * (
                    np.linalg.norm(Trial_u[:, j])
                ) ** 2 + c * np.absolute(Trial_G[0][j])
                Merit_new[j] = merit_new

            trial_step_size = Trial_step_size[0][0]
            merit_new = Merit_new[0]

            j = 0

            while merit_new > merit and j < ntrial:
                trial_step_size = Trial_step_size[0][j]
                merit_new = Merit_new[j]
                j += 1
                if j == ntrial and merit_new > merit:
                    if self.options.getPrintOutput():
                        print(
                            "The step size has been reduced by a factor of 1/",
                            2**ntrial,
                        )
        step_size = trial_step_size
        return step_size

    def computeBeta(self):
        """Compute beta value"""
        self.beta = np.dot(self.alpha, self.u)

    def computeFailureProbability(self):
        """Compute probability of failure"""
        self.Pf = normal.cdf(-self.beta)

    def showResults(self):
        """Show results"""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        n_hyphen = 54
        print("")
        print("=" * n_hyphen)
        print("")
        print(" RESULTS FROM RUNNING FORM RELIABILITY ANALYSIS")
        print("")
        print(" Number of iterations:     ", self.i)
        print(" Reliability index beta:   ", self.beta[0])
        print(" Failure probability:      ", self.Pf[0])
        print(
            " Number of calls to the limit-state function:",
            self.getNoFunctionCalls(),
        )
        print("")
        print("=" * n_hyphen)
        print("")

    def showDetailedOutput(self):
        """Get detailed output to console"""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        names = self.model.getVariables().keys()
        consts = self.model.getConstants()
        u_star = self.getDesignPoint()
        x_star = self.getDesignPoint(uspace=False)
        alpha = self.getAlpha()

        n_hyphen = 54
        print("")
        print("=" * n_hyphen)
        print("FORM")
        print("=" * n_hyphen)
        print("{:15s} \t {:1.10e}".format("Pf", self.Pf[0]))
        print("{:15s} \t {:2.10f}".format("BetaHL", self.beta[0]))
        print(
            "{:15s} \t {:d}".format("Model Evaluations", self.model.getCallFunction())
        )
        print("-" * n_hyphen)
        print(
            "{:10s} \t {:>9s} \t {:>12s} \t {:>9s}".format(
                "Variable", "U_star", "X_star", "alpha"
            )
        )
        for i, name in enumerate(names):
            print(
                "{:10s} \t {: 5.6f} \t {:12.6f} \t {:+5.6f}".format(
                    name, u_star[i], x_star[i], alpha[i]
                )
            )
        for name, val in consts.items():
            print(f"{name:10s} \t {'---':>9s} \t {val:12.6f} \t {'---':>9s}")
        print("=" * n_hyphen)
        print("")

    def getBeta(self):
        """Returns the beta value

        :Returns:
          - beta (float): Returns the beta value
        """
        return self.beta[0]

    def getFailure(self):
        """Returns the probability of failure

        :Returns:
          - Pf (float): Returns the probability of failure
        """
        return self.Pf

    def getDesignPoint(self, uspace=True):
        """Returns the design point, defaults to u-space

        :Returns:
          - u (float): Returns the design point in u- or x-space
        """
        if uspace:
            return self.u
        else:
            return self.transform.u_to_x(self.u, self.model.getMarginalDistributions())

    def getAlpha(self, as_dict=False):
        """Returns the alpha vector

        :Returns:
          - alpha (np.array): Returns the alpha vector
        """
        if as_dict:
            names = self.model.getNames()
            alphas = self.alpha[0]
            alpha_dict = {name: alpha for alpha, name in zip(alphas, names)}
            return alpha_dict
        return self.alpha[0]

    def getNoFunctionCalls(self):
        """
        Returns the number of function evaluations used

        :Returns:
          - n (int): Returns the number of function evaluations

        """
        return self.model.getCallFunction()
