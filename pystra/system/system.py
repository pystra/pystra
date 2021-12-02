#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product, combinations
from scipy.stats import multivariate_normal, norm

from ..model import *
from ..form import *


class Component():
    """
    A component within a system described by a limit state
    """

    def __init__(self, name, limit_state, analysis_options=None):
        """
        Parameters
        ----------
        name : string
            description of component
        limit_state : LimitState class object
            Information about the limit state
        analysis_option: AnalysisOption class object
            Information about the analysis options.
            Defaults when not provided.
        """
        self.name = name
        self.limit_state = limit_state
        self.stochastic_model = StochasticModel()
        # Also instance a stochastic model

        if analysis_options is None:
            self.options = AnalysisOptions()
            # if not specified, using default for now,
        else:
            self.options = analysis_options

        self.event_vector = [1, 0]
        # used to indicate failure or not

    def addVariable(self, obj):
        """
        Add variable for stochastic model of component
        """
        self.stochastic_model.addVariable(obj)

    def setCorrelation(self, obj):
        """
        Define correlation for stochastic model of component
        """
        self.stochastic_model.setCorrelation(obj)

    def setLimitState(self, limit_state):
        """
        Change limit state for component

        Parameters
        ----------
        limit_state : LimitState class object
            Information about the limit state
        """
        self.limit_state = limit_state

    def setOptions(self, analysis_options):
        """
        Change analysis options for component

        Parameters
        ----------
        analysis_option: AnalysisOption class object
            Information about the analysis options.
        """
        self.options = analysis_options

    def getProbability(self):
        """
        Obtain the beta and alpha using FORM for component
        """
        analysis = Form(
            analysis_options=self.options,
            stochastic_model=self.stochastic_model,
            limit_state=self.limit_state,
        )
        beta = float(analysis.beta)
        pf = analysis.Pf

        # Get unique variable names for alphas (used for alpha_hat)
        variable_list = list(self.stochastic_model.variables.keys())
        arg_names = [s for s in variable_list]
        # Make dataframe for alphas to concat with others later
        alphas = pd.DataFrame(analysis.alpha, columns=arg_names)

        return beta, alphas, pf


class System():
    """
    Abstract base class for a collection of components
    (either in series/paralel/both)
    """

    def __init__(self, obj_list):

        self.components = []  # list of components objs
        self.event_vectors = []  # list of individual event_vectors
        self.event_vector = None  # specific system event vectors
        # used to describe which MECE components events occurs in event
        self.rvs_autocorrelation_matrix = None
        # autocorrelation matrix between random variables
        self.cmp_correlation_matrix = None
        # correlation matrix between components

        self.addComponents(obj_list)
        # populate component list

        # indicator if a mixed-system
        self.ismixed = len(obj_list) < len(self.components)

    def addComponents(self, obj_list):
        """Append component objects to system list."""

        for obj in obj_list:  # loop through the objects

            if isinstance(obj, System):  # if the object is a system
                self.components += obj.components
                # append the components of that system
            else:  # if the object is a component
                self.components.append(obj)
                # append the component

            self.event_vectors.append(obj.event_vector)
            # also append the event vector for component/system

        # now find the combo array used to find the event vector for system
        self.combo_array = np.array(list(product(*self.event_vectors)))

    def setCorrelation(self, cm_obj, of_components=False):
        """
        Define the correlation in system
        (either for auto correlation variables or correlation of componets)

        Parameters
        ----------
        cm_obj : array
            correlation matrix in correct order.
            1D for autocorrelation. 2D for component correlation.
        of_components : bool, optional
            whether the correlation of the system components
            rather than of the autocorrelation of variables.
            The default is False. Correlation of components is then calculated.
        """

        # Set correlation matrix
        if of_components:
            self.cmp_correlation_matrix = cm_obj
        else:
            self.rvs_autocorrelation_matrix = np.diagflat(cm_obj)

    def eventcorrelations(self, alpha_hat):
        """Convert correlations between random variables to correlations
        between component events (limit states), if not specified"""

        a = alpha_hat
        pk = self.rvs_autocorrelation_matrix
        if self.cmp_correlation_matrix is None:
            self.cmp_correlation_matrix = np.dot(np.dot(a, pk), a.T)
        # and now replace the diagonal with ones
        # (correlation between same limit state)
        np.fill_diagonal(self.cmp_correlation_matrix, 1)

    def getBounds(self, method="default"):
        """
        Obtain system bounds. Not accepted for mixed systems.

        Parameters
        ----------
        method : string, optional
            Method used. "default" is simple bounds.

        Returns
        -------
        bounds: tuple

        """
        if self.ismixed:
            raise TypeError("Cannot provide bounds for mixed system")
        else:
            (lower, upper) = eval("self.bounds_" + method + "()")
            return (lower, upper)

    def getReliability(self, method="default"):
        """
        Obtains system reliability,
        expressing as betas and probability of failure
        """
        # event vector obtained through init
        self.getEventProbabilities(method)
        self.system_probability = np.sum(
            self.event_probabilities * self.event_vector)
        system_beta = -norm.ppf(self.system_probability)
        return system_beta, self.system_probability

    def componentprobabilities(self):
        """
        Calculate and populate multiple component reliabilties
        and their correlation
        """

        betas = []  # Pre-allocated beta list of floats
        pfs = []  # Pre-allocated beta list of floats
        alphas = []  # Pre-allocated alpha list of dataframes

        for component in self.components:
            # Run individual independent FORM analysis
            beta, alpha, pf = component.getProbability()

            betas.append(beta)  # Append
            pfs.append(pf)  # Append
            alphas.append(alpha)

        alpha_hat = pd.concat(alphas).fillna(0)
        # get the alpha matrix for system

        # Get the random variable order for correlation matirx
        self.random_variables = list(alpha_hat.columns)
        # unique random variables
        n_rvs = len(self.random_variables)

        # Convert to arrays
        betas = np.array(betas)
        alpha_hat = np.array(alpha_hat)
        pfs = np.array(pfs)

        # Correlation
        if self.rvs_autocorrelation_matrix is None:  # set default correlation
            # number of random variables in system
            self.rvs_autocorrelation_matrix = np.eye(n_rvs)

        self.eventcorrelations(alpha_hat)  # find the component correlations

        return betas, alpha_hat, pfs

    def getEventProbabilities(self, method="default"):
        """
        Obtains the probability failure of each MECE event
        for different methods
        """

        betas, _, _ = self.componentprobabilities()

        if method == "default":
            self.eventprobabilities_default(betas)
        else:
            raise ValueError("Only default method currently supported")

    def eventprobabilities_default(self, betas):
        """
        Obtains the probability failure of each MECE event with
        first-order system reliability

        """

        # Get component probabilities

        # Enumerating MECE events
        n = len(betas)
        sign = [1, -1]  # 1 for fail, -1 for success
        combs = pd.Series(list(product(sign, repeat=n)))

        # Function for database apply for probabilities
        def event_prob(event_comb, b, v):
            z1 = np.zeros((len(event_comb), len(event_comb)))
            z2 = np.zeros((len(event_comb), len(event_comb)))
            z1[:, 0] = event_comb
            z2[0, :] = event_comb
            # Needed to make matrix multipication possible
            row = np.array(event_comb)
            return multivariate_normal.cdf(
                row * -b, cov=np.dot(z1, z2) * v, allow_singular=True
            )
            # allow_singular provides nearest positive definite

        # database mapping
        event_probs = combs.apply(event_prob,
                                  b=betas, v=self.cmp_correlation_matrix)
        # convert to list
        self.event_probabilities = list(event_probs)


class SeriesSystem(System):
    """
    A collection of components (or systems) in series system
    """

    def __init__(self, obj_list):
        """
        Parameters
        ----------
        obj_list : list
            List of components (or system) arranged in a series system

        """
        super().__init__(obj_list)
        self.event_vector = self.getEventVector()

    def getEventVector(self):
        """
        Establish the event vector for series system as
        1 - (1 - e1)*(1 - e2)....(1 - en)
        where e is the component/system event (and columns of combo_array)
        """
        one_array = np.ones_like(self.combo_array)
        # create an ones array size of combo_array
        one_vector = np.ones_like(self.combo_array[:, 1])
        # create an ones vector length of one column from combo_array

        return one_vector - np.prod((one_array - self.combo_array), axis=1)
        # calculate series system event vector

    def bounds_default(self):
        "simple bounds for series system"

        # get component probabilities
        _, _, pfs = self.componentprobabilities()

        lower = np.max(pfs)  # lower bound
        upper = 1 - np.prod(1 - pfs)  # upper bound

        return lower, upper

    def bounds_ditlevsen(self):
        "ditlevsen bounds for series system"

        # get component probabilities
        betas, _, pfs = self.componentprobabilities()

        pfs[::-1].sort()  # sort in descending
        betas.sort()  # sort in ascending

        # get selected joint probabilties
        indices = range(0, len(betas))
        combs = list(combinations(indices, 2))  # get unique indicies pairs

        joint_probs = np.zeros((len(betas), len(betas)))  # preallocated

        for c in combs:

            # correlation
            v = self.cmp_correlation_matrix[c]
            cov_m = np.array([[1, v], [v, 1]])

            # bivariate calculation of unique pairs
            joint_probs[c] = multivariate_normal.cdf(
                -betas[list(c)], cov=cov_m, allow_singular=True
            )

        # upper bound
        upper = np.sum(pfs) - np.sum(np.max(joint_probs, axis=0))

        # lower bound
        lower = float(pfs[0]) + np.sum(
            np.maximum(pfs[1:].T - np.sum(joint_probs, axis=0)[1:], 0)
        )

        return lower, upper


class ParallelSystem(System):
    """
    A collection of components (or systems) in parallel system
    """

    def __init__(self, obj_list):
        """
        Parameters
        ----------
        obj_list : list
            List of components (or system) arranged in a series system

        """
        super().__init__(obj_list)
        self.event_vector = self.getEventVector()

    def getEventVector(self):
        """
        Establish the event vector for parallel system as
        (e1)*(e2)....(en)
        where e is the component/system event (and columns of combo_array)
        """
        return np.prod(self.combo_array, axis=1)
        # calculate parallel system event vector

    def bounds_default(self):
        "simple bounds for parallel system"

        # get component probabilities
        _, _, pfs = self.componentprobabilities()

        upper = np.min(pfs)  # upper bound
        lower = np.prod(pfs)  # lower bound

        return lower, upper
