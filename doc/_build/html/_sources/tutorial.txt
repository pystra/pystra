.. _chap_tutorial:

********
Tutorial
********

This tutorial will guide you through a typical PyMC application. Familiarity
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

here we must write for each command ``pyre.my_command()``. A much nicer way to
load the package is case 2: ::

  # import pyre library
  from pyre import *

here, we import all available objects from ``pyre``.

To define the random variables from :eq:`random_variables` we can use
following syntax: ::

  # Define random variables
  X1 = Lognormal('X1',500,100)
  X2 = Normal('X2',2000,400)
  X3 = Uniform('X3',5,0.5)

The first parameter is the name of the random variable. The name has to be a
string, so the input looks like ``'X3'``.

By default, the next to values are the first and second moment of the
distribution, here mean and standard deviation. Are mean and standard
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

In the same way, we can add the correlation matrix to our model: ::

  # Define Correlation Matrix
  Corr = CorrelationMatrix([[1.0, 0.3, 0.2],
                            [0.3, 1.0, 0.2],
                            [0.2, 0.2, 1.0]])

Are the variables uncorrelated, you don't have to add a correlation matrix to
the model.

At least we have to define the limit state function. Therefore are two ways:

* Direct in the ``main`` code,
* in a separate ``function``.

In the first case the input will look like: ::

  # Define limit state function
  # - case 1: define directly
  g = LimitStateFunction('1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2')

and in the second case like this: ::

  # Define limit state function
  # - case 2: define load function, wich is defined in function.py
  g = LimitStateFunction('function(X1,X2,X3)')

The function ``'function(X1,X2,X3)'`` can be found in ``'function.py'``. This
case can be useful if the limit state function is quiet complex or need more
then one line to define it. Here ``'function.py'`` is defined as: ::

  def function(X1,X2,X3):
    g = 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2
    return g

At this stage our model is complete defined and we can start the analysis.


Reliability Analysis
--------------------

To store the results from the analysis an object must be initialized: ::

  # Performe FORM analysis
  Analysis = Form()

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

To change some options, a object must be initialized which stores the
customized options. ::

  # Set some options (optional)
  options = AnalysisOptions()
  options.printResults(False)

and This options must be implemented in our analysis: ::

  # Performe FORM analysis
  Analysis = Form(options)

To use the results for further calculations, plots etc. the results can get by
some getter methods (see :ref:`chap_calculations`) ::

  # Some single results:
  beta = Analysis.getBeta()
  failure = Analysis.getFailure()

Finally...
----------

This was a short introduction how to use ``pyre``. The tutorial above is also
available on `GitHub`_ under ``example.py``.

Let's have fun ;)

.. _`Python documentation`: http://www.python.org/doc/

.. _`GitHub`: https://github.com/hackl/pyre
