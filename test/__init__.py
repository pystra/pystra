# -*- coding: utf-8 -*-
"""

@author: heijer

"""
import unittest

# import distributions
from pystra.distributions import Normal, Lognormal, Uniform

# import reliability methods
from pystra.form import Form
from pystra.mc import CrudeMonteCarlo, ImportanceSampling

# import model helpers
from pystra import model, correlation


def example_limitstatefunction(X1, X2, X3):
    """
    example limit state function
    """
    return 1 - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2


class UnitTests(unittest.TestCase):
    def setUp(self):
        """
        Set up simulation
        """
        # Define limit state function
        # - case 1: define directly as lambda function
        # self.g = lambda X1,X2,X3: 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2
        # - case 2: use predefined function
        self.g = example_limitstatefunction

        # case 3 and 4 are NOT RECOMMENDED but still available for backward compatibility
        # - case 3: define directly as string expression
        # self.g = '1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2'
        # - case 4: use function from function.py, expression as string
        # self.g = 'function(X1,X2,X3)'

        # Set some options (optional)
        self.options = model.AnalysisOptions()
        self.options.printResults(False)
        self.options.setSamples(1000)  # only relevant for Monte Carlo

        # Set stochastic model
        self.stochastic_model = model.StochasticModel()
        # Define random variables
        self.stochastic_model.addVariable(Lognormal("X1", 500, 100))
        self.stochastic_model.addVariable(Normal("X2", 2000, 400))
        self.stochastic_model.addVariable(Uniform("X3", 5, 0.5))

        self.stochastic_model.setCorrelation(
            correlation.CorrelationMatrix(
                [[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]]
            )
        )

        # Set limit state
        self.limit_state = model.LimitState(self.g)

    def test_form(self):
        """
        Perform FORM analysis
        """
        Analysis = Form(
            analysis_options=self.options,
            stochastic_model=self.stochastic_model,
            limit_state=self.limit_state,
        )

        # validate results
        # self.assertEqual(Analysis.i, 17)
        # self.assertAlmostEqual(Analysis.beta, 1.75, places=2)

        # print beta
        # print 'FORM', 'beta:', Analysis.getBeta()

    def test_cmc(self):
        """
        Perform Crude Monte Carlo Simulation
        """
        Analysis = CrudeMonteCarlo(
            analysis_options=self.options,
            stochastic_model=self.stochastic_model,
            limit_state=self.limit_state,
        )

        # validate results
        # self.assertEqual(Analysis.x.shape[-1], 1000)

        # print beta
        # print 'CMC', 'beta:', Analysis.getBeta()

    def test_is(self):
        """
        Perform Importance Sampling
        """
        Analysis = ImportanceSampling(
            analysis_options=self.options,
            stochastic_model=self.stochastic_model,
            limit_state=self.limit_state,
        )

        # validate results
        # self.assertEqual(Analysis.x.shape[-1], 1000)

        # print beta
        # print 'IS', 'beta:', Analysis.getBeta()


if __name__ == "__main__":
    unittest.main()
