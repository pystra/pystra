# -*- coding: utf-8 -*-
"""

@author: heijer

"""
import unittest
from pyre import *

class UnitTests_form(unittest.TestCase):
    def setUp(self):
        """
        Set up simulation
        """
        # Define random variables
        self.X1 = Lognormal('X1',500,100)
        self.X2 = Normal('X2',2000,400)
        self.X3 = Uniform('X3',5,0.5)
        # Define limit state function
        # - case 1: define directly
        self.g = LimitStateFunction('1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2')
        
        # Set some options (optional)
        self.options = AnalysisOptions()
        self.options.printResults(False)
        self.options.setSamples(1000) # only relevant for Monte Carlo
        
        # Set stochastic model
        self.stochastic_model = StochasticModel()
        
        # Set limit state
        self.limit_state = LimitState()
    def test_form(self):
        """
        Perform FORM analysis
        """
        Analysis = Form(analysis_options=self.options,
                        stochastic_model=self.stochastic_model,
                        limit_state=self.limit_state)

        # validate results
        self.assertEqual(Analysis.i, 13)
        self.assertAlmostEqual(Analysis.beta, 1.65, places=2)

        # print beta
        print 'FORM', 'beta:', Analysis.getBeta()
    def test_cmc(self):
        """
        Perform Crude Monte Carlo Simulation
        """
        Analysis = CrudeMonteCarlo(analysis_options=self.options,
                        stochastic_model=self.stochastic_model,
                        limit_state=self.limit_state)

        # validate results
        self.assertEqual(Analysis.x.shape[-1], 1000)

        # print beta
        print 'CMC', 'beta:', Analysis.getBeta()
    def test_is(self):
        """
        Perform Importance Sampling
        """
        Analysis = ImportanceSampling(analysis_options=self.options,
                        stochastic_model=self.stochastic_model,
                        limit_state=self.limit_state)

        # validate results
        self.assertEqual(Analysis.x.shape[-1], 1000)

        # print beta
        print 'IS', 'beta:', Analysis.getBeta()

if __name__ == '__main__':
    unittest.main()