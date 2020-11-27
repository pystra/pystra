.. _chap_tutorial:

********
Tutorial
********

This tutorial will guide you through a typical PyRe application. Familiarity
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

Additionaly those variables are relatet to each other. Therefore the
correlationmatrix :math:`{\bf C}` is given:

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
probability :math:`P_f`, by given limit state function :math:`g(X_1,X_2,X_3)`:

.. math::
    :label: limit_state_function

            g(X_1,X_2,X_3) = 1 - \frac{X_2}{1000 \cdot X_3} - 
            \left( \frac{X_1}{200 \cdot X_3} \right)^2

Let's model
-----------

Before we start with the modeling, we have to import the ``pyre``
package. Therefore are two different methods available:

In case 1 we load ``pyre`` like a normal library: ::

  import pyre

here we must write for each command ``pyre.my_command()``. A shorter way to
load the package is case 2: ::

  # import pyre library
  from pyre import *

here, we import all available objects from ``pyre``.

Two ways to define the limit state function are available: 

* Direct in the ``main`` code,
* as a separate ``function``.

In the first case the input will look like: ::

  # Define limit state function
  # - case 1: define directly
  limit_state = LimitState(lambda X1,X2,X3: 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2)

and in the second case like this: ::

  # Define limit state function
  # - case 2: use predefined function
  limit_state = LimitState(example_limitstatefunction)

The function ``example_limitstatefunction`` has be defined in advance
as a seperate function such as:::

  def example_limitstatefunction(X1,X2,X3):
      return 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2

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
  # options.printResults(False)

To store the results from the analysis an object must be initialized: ::

  # Performe FORM analysis
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
``printResults(False)`` has set to be ``False``. There are also some other
options which can be modified (see :ref:`chap_model`).

To use the results for further calculations, plots etc. the results can get by
some getter methods (see :ref:`chap_calculations`) ::

  # Some single results:
  beta = Analysis.getBeta()
  failure = Analysis.getFailure()

There is also the possibility to output more detailed results using 
``showDetailedOutput()``: ::

  =====================================================
  FORM
  =====================================================
  Pf                       0.0397172978
  BetaHL                   1.7539761407
  Model Evaluations        164
  -----------------------------------------------------
  Variable            U_star             X_star        alpha
  X1                1.278045         631.504135    +0.728414
  X2                0.407819        2310.352495    +0.232354
  X3               -1.129920           4.517374    -0.644534
  =====================================================

Finally...
----------

This was a short introduction how to use ``pyre``. The tutorial above is also
available on `GitHub`_ under ``example.py``.

Let's have fun ;)

.. _`Python documentation`: http://www.python.org/doc/

.. _`GitHub`: https://github.com/hackl/pyre
