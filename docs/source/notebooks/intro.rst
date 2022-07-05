*********************
Introductory Turorial
*********************

This tutorial will guide you through a typical Pystra application. Familiarity
with Python is assumed, so if you are new to Python, books such as [Lutz2007]_
or [Langtangen2009]_ are the place to start. Plenty of online documentation
can also be found on the `Python documentation`_ page.


An example reliability model
----------------------------

Consider the following random variables:

.. math::
    :label: random_variables

            \begin{align}
            X_1 &\sim \text{Logormal}(500,100)\\
            X_2 &\sim \text{Normal}(2000,400)\\
            X_3 &\sim \text{Uniform}(5,0.5)
            \end{align}

Additionally those variables are related to each other. Therefore the
correlation matrix :math:`{\bf C}` is given:

.. math::
    :label: correlation_matrix

            \begin{align}
            {\bf C} = 
            \begin{pmatrix}
            1.0 & 0.3 & 0.2\\
            0.3 & 1.0 & 0.2\\
            0.2 & 0.2 & 1.0
            \end{pmatrix}
            \end{align}

Now, we like to compute the reliability index :math:`\beta` and the failure
probability :math:`P_f`, by given limit state function :math:`g(\gamma, X_1,X_2,X_3)`:

.. math::
    :label: limit_state_function

            g(\gamma X_1,X_2,X_3) = \gamma - \frac{X_2}{1000 \cdot X_3} - 
            \left( \frac{X_1}{200 \cdot X_3} \right)^2

where :math:`\gamma` is a real constant. For this example, let :math:`\gamma = 1`.

Let's model
-----------

Before we start with the modeling, we have to import the ``pystra``
package. Therefore are two different methods available:

In case 1 we load ``pystra`` like a normal library: ::

  import pystra

here we must write for each command ``pystra.my_command()``. A shorter way to
load the package is case 2: ::

  # import pystra library
  from pystra import *

here, we import all available objects from ``pystra``.

Two ways to define the limit state function are available: 

* Direct in the ``main`` code,
* as a separate ``function``.

In the first case the input will look like: ::

  # Define limit state function
  # - case 1: define directly
  limit_state = LimitState(lambda g,X1,X2,X3: g - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2)

and in the second case like this: ::

  # Define limit state function
  # - case 2: use predefined function
  limit_state = LimitState(example_limitstatefunction)

The function ``example_limitstatefunction`` has be defined in advance
as a separate function such as:::

  def example_limitstatefunction(g,X1,X2,X3):
      return g - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2

This case can be useful if the limit state function is quiet complex
or need more then one line to define it.

In the next step the stochastic model has to be initialized ::

  stochastic_model = StochasticModel()

and the random variables have to be assigned. To define the random
variables from :eq:`random_variables` we can use following syntax: ::

  # Define random variables
  stochastic_model.addVariable( Lognormal('X1',500,100) )
  stochastic_model.addVariable( Normal('X2',2000,400) )
  stochastic_model.addVariable( Uniform('X3',5,0.5) )

The first parameter is the name of the random variable. The name has to be a
string and match the arguments in the limit state function, so the input looks like ``'X3'``.

By default, the next to values are the first and second moment of the
distribution, here mean and standard deviation. If mean and standard
deviation unknown but the distribution parameter known, then the
``input_type`` has to be changed.

For example random variable :math:`X_3` is uniform distributed. Above we
assume that :math:`X_3` is defined by mean and standard deviation. But we can
describe the distribution with the parameter :math:`a` and :math:`b`. In this
case the code will look like: ::

  X3 = Uniform('X3',4.133974596215562, 5.866025403784438, 1)

to get the same results as before. To see which parameters are needed and in
which order the must insert, take a look at Chapter :ref:`chap_distributions`.
There are all currently available distributions listed.

If the nominal value, bias, and coefficient of variation are instead known,
then the random variable can be instantiated following this example: ::

  X2 = Normal('X2',*500*1.00*np.array([1, 0.2]))

where nominal value is 500, bias is 1.00, and coefficient of variation is 0.2. 
Notice the initial `*` character is used to dereference the output array.

We will also define our constant using ``Constant``:  ::

  # Define constants
  stochastic_model.addVariable( Constant('g',1) )


To add the correlation matrix to our model: ::

  # Define Correlation Matrix
  stochastic_model.setCorrelation( CorrelationMatrix([[1.0, 0.3, 0.2],
                                                      [0.3, 1.0, 0.2],
                                                      [0.2, 0.2, 1.0]]) )

If the variables uncorrelated, you don't have to add a correlation matrix to
the model.

At this stage our model is complete defined and we can start the analysis.

Reliability Analysis
--------------------

To change some options, a object must be initialized which stores the
customized options. ::

  # Set some options (optional)
  options = AnalysisOptions()
  # options.setPrintOutput(False)

To store the results from the analysis an object must be initialized: ::

  # Perform FORM analysis
  Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)

Now the code can be compiled and the FORM analysis will be preformed. In this
example we will get following results: ::

  ==================================================

   RESULTS FROM RUNNING FORM RELIABILITY ANALYSIS

   Number of iterations:      17
   Reliability index beta:    1.75397614074
   Failure probability:       0.039717297753
   Number of calls to the limit-state function: 164

  ==================================================

If we don't like to see the results in the terminal the option
``setPrintOutput(False)`` has set to be ``False``. There are also some other
options which can be modified (see :ref:`chap_model`).

To use the results for further calculations, plots etc. the results can get by
some getter methods (see :ref:`chap_calculations`) ::

  # Some single results:
  beta = Analysis.getBeta()
  failure = Analysis.getFailure()

There is also the possibility to output more detailed results using 
``showDetailedOutput()``: ::

  ======================================================
  FORM
  ======================================================
  Pf              	 3.9717297753e-02
  BetaHL          	 1.7539761407
  Model Evaluations 	 164
  ------------------------------------------------------
  Variable   	    U_star 	       X_star 	     alpha
  X1         	  1.278045 	   631.504135 	 +0.728414
  X2         	  0.407819 	  2310.352495 	 +0.232354
  X3         	 -1.129920 	     4.517374 	 -0.644534
  g          	       --- 	     1.000000 	       ---
  ======================================================

A Second-Order Reliability Method (SORM) can also be performed, passing in the 
results of a FORM analysis object if it exists, fo; efficiency (otherwise, SORM
will perform a FORM analysis first): ::

    sorm = Sorm(analysis_options=options,stochastic_model=stochastic_model, 
                limit_state=limit_state, form=Analysis)
    sorm.run()

for the example, this produces the output: ::

  ======================================================
  
  RESULTS FROM RUNNING SECOND ORDER RELIABILITY METHOD
  
  Generalized reliability index:  1.8489979688766982
  Probability of failure:         0.0322290530029448
  
  Curavture 1: -0.04143130882429444
  Curavture 2: 0.36356407501548915
  ======================================================

Similar to FORM, we can also get more detailed output for diagnostics: ::

    sorm.showDetailedOutput()

which for the example gives: ::

  ======================================================
  FORM/SORM
  ======================================================
  Pf FORM         		 3.9717297753e-02
  Pf SORM Breitung 		 3.2229053003e-02
  Pf SORM Breitung HR 	 3.1158626124e-02
  Beta_HL         		 1.7539761407
  Beta_G Breitung 		 1.8489979689
  Beta_G Breitung HR 	 1.8640317040
  Model Evaluations 	 180
  ------------------------------------------------------
  Curvature 1: -0.04143130882429444
  Curvature 2: 0.36356407501548915
  ------------------------------------------------------
  Variable   	    U_star 	       X_star 	     alpha
  X1         	  1.278045 	   631.504135 	 +0.728414
  X2         	  0.407819 	  2310.352495 	 +0.232354
  X3         	 -1.129920 	     4.517374 	 -0.644534
  g          	       --- 	     1.000000 	       ---
  ======================================================

in which `HR` refers to the Hohenbichler-Rackwitz modification to Breitung's
formula.

Finally...
----------

This was a short introduction how to use ``pystra``. The tutorial above is also
available on `GitHub`_ under ``example.py``.

Let's have fun ;)

.. _`Python documentation`: http://www.python.org/doc/

.. _`GitHub`: https://github.com/pystra/pystra
