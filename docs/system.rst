.. _chap_system:

********
System Reliability
********

Pystra can perform system reliability analysis.
A system comprises of two or more components either a series (weak-link, chain) or parallel (all-or-nothing) configurations.
Mixed systems (series and parallel) are also accepted.


Usage
==================

Defining Components
-------------------

Each component in the system is assigned a name and `LimitState()` variable. Specific options for analysis may also be added. ::

  # import pystra library
  import pystra as pr
  
  # Define limit state function
  limit_state = pr.LimitState(lambda X1,X2: X1 - X2)
  
  # Define a component (using default options)
  comp = pr.Components("A",limit_state,pr.AnalysisOptions())

Variables and their correlations (if any) for the component are defined in similar manner to a component reliability analysis.
However, rather than `stochastic_model`, the instantiated component is used: ::

  # Add a random variable
  comp.addVariable(pr.Lognormal('X1',500,100))
  
  # Add a correlation matrix
  comp.setCorrelation( pr.CorrelationMatrix([[1.0, 0.3],
                                          [0.3, 1.0]]) )

Once each component is defined, it is possible to run a FORM analysis for each component by calling the ``.getProbability()`` method.


Defining a System
------------------

Assuming three components have been instantiated, to define a series system: ::

	series_sys = pr.SeriesSystem([comp1,comp2,comp3])

or to define a parallel system: ::

	parallel_sys = pr.ParallelSystem([comp1,comp2,comp3])

A mixed-system is defined as: ::

	serpar_sys = pr.SeriesSystem([comp1,
								  pr.ParallelSystem([comp2,comp3])])


The system reliability index and probability of failure is then obtained by calling the ``.getReliability()`` method.


Component Correlations and Autocorrelation
------------------------------------------

When a random variable is defined with the same name between two different components, 
by default it is taken that this is the same random variable.
That is, its autocorrelation is 1. 
Consequently, a correlation between components that share this random variable exists.
The system reliability is calculated noting any correlation between components.

If the autocorrelation of a random variable is 0 (ie. independent variables, although defined with the same name),
or another value, the correlation is set for the system as: ::

	series_sys.setCorrelation(np.array([0,0]))

where the list values is the autocorrelation of the random variable in order as they were instantiated.

.. note::

   Autocorrelations not equal to 0 and 1 have not been validated.


Theoretical Background
==================

The system reliability analysis is based on the Matrix-Based System Reliability (MSR) Method [Kang2008]_.
Consider a systems event with *n* components, assuming each component has two
distinct states, i.e. failure and survival, the sample space can be subdivide into :math:`v = 2^n` and the failure
mutually exclusive and collectively exhaustive (MECE) “basic” events, :math:`e_j`, :math:`j = 1, … , v`.
Then, any system event can be represented by an “event” column vector :math:`\mathbf{c}` whose *j*-th
element is 1 if :math:`e_j` belongs to the system event and 0 otherwise. Since the basic events are
mutually exclusive the probability of the system event, :math:`P(E_{sys})`, is the sum of the
probabilities of basic events that belong to the system event. Therefore, the system
probability is computed by the inner product of the two vectors:

.. math::
    :label: msr

            P(E_{sys}) = \mathbf{c}_E^{T}\mathbf{p}

where :math:`\mathbf{p}` is the probability column vector that contains :math:`p_j = P(e_j)`, 
:math:`j = 1, … , v`.

Establishing the Event Vector
-----------------------------

Let :math:\mathbf{c}_i = [1 , 0] be the event vector for component *i*, describing the state (i.e. failure and survival).
For a system with *n* components, the collection of MECE events is enumerated as the Cartesian project of :math:\mathbf{c}_i *n*-times.
In this way, each MECE event :math:\mathbf{c}_j is described by each of the rows of the resulting array as 1 or 0 with length *n*.

To find a system event vector :math:\mathbf{c}_E, it follows that: 

.. math::
    :label: correlation_matrix

            \begin{align}
			\mathbf{c}_\bar{E} & = \mathbf{1} - \mathbf{c}_{E} \\
			\mathbf{c}_{E_{parallel}} & = \prod \mathbf{c}_{j}} \\
			\mathbf{c}_{E_{series}} & = \mathbf{1} - \prod (1-\mathbf{c}_{j}}) \\
            \end{align}
where :math:\mathbf{1} is a vector of ones that has the same size as the event vector 
and * indicates element-by-element matrix multiplication.

Note that in a mixed-system, these equations are still valid, with the system event vector :math:\mathbf{c}_E used rather than the MECE vector :math:\mathbf{c}_j. 

Quantifying MECE Probabilities
-------------------------------

By default, the probabilities of each MECE event is calculated 
using a procedure extending from the equivalent planes method (EPM) [Roscoe2015]_
and the sequential compounding method (SCM) [Kang2010]_.

.. math::
    :label: msr_probs
            p_j \approx \Phi_N\left(-[s]*\boldsymbol{\beta},([s]^T \cdot [s])*\boldsymbol{P}\right)


where :math:[s] indicates the row vector describing the MECE event using 1 and/or -1 notation (rather than 0), 
* indicates element-by-element matrix multiplication, :math:\cdot indicates matrix multiplication, 
:math:\boldsymbol{\beta} is a vector of the component reliability indicies (found using FORM),
:math:\boldsymbol{P} is the covariance matrix describing the correlation between components with entries found as:

.. math::
    :label: rho_calc
            \rho_{ab} = \sum_{k=1}^{V} \alpha_{a,k}\alpha_{b,j} \rho_{ab,k}

with :math:\alpha_{i,k} is the influence coefficient (directional cosine) for random variable *k* in the component *i* (found using FORM),
and :math:\rho_{ab,k} the autocorrelation of the random variable *k*.
 
If $\boldsymbol{P}$ is singular due to correlation values, then closest variance-covariance matrix is sought.

.. note::
   Alternative procedures such as Monte Carlo simulation (MCS) are yet to be explored.

