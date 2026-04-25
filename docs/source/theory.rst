**********************
Theoretical Background
**********************

Structural Reliability
======================


Structural reliability analysis (SRA) is an important part to handle
structural engineering applications [Melchers1999]_. This section provides
a brief introduction to this topic and is also the theoretical background
for the Python library, Python Structural Reliability Analysis (`Pystra`).

Limit States
------------
The word structural reliability refers to the meaning "how
much reliable is a structure in terms of fulfilling its
purpose" [Malioka2009]_. The performance of structures and engineering
systems was based on deterministic parameters even for a long time, even if it
was known that all stages of the system involve uncertainties. SRA provides a
method to take those uncertainties into account in a consistent manner. In
this content the term probability of failure is more common than
reliability. [Malioka2009]_

In general, the term "failure" is a vague definition because it means
different things in different cases. For this purpose the concept of limit
state is used to define failure in the context of SRA. [Nowak2000]_

.. note::

   A limit state represents a boundary between desired and undesired
   performance of a structure.

This boundary is usually interpreted and formulated within a mathematical
model for the functionality and performance of a structural system, and
expressed by a limit state function. [Ditlevsen2007]_

.. note::
   [Limit State Function]

   Let :math:`{\bf X}` describe a set of random variables :math:`{X}_1
   \dots {X}_n` which influence the performance of a structure. Then the
   functionality of the structure is called limit state function, denoted by
   :math:`g` and given by

   .. math::
      :label: eq:2_69

              g({\bf X})=g(X_1,\dots,X_n)

The boundary between desired and undesired performance would be given when
:math:`g({\bf X}) = 0`. If :math:`g({\bf X}) > 0`, it implies a desired
performance and the structure is safe. An undesired performance is given by
:math:`g({\bf X}) \leq 0` and it implies an unsafe structure or failure of the
system. [Baker2010]_

The probability of failure :math:`p_f` is equal to the probability that an
undesired performance will occur. It can be mathematical expressed as

.. math::
   :label: eq:2_70

           p_f = P(g({\bf X})\leq 0) = \iiint\limits_{g({\bf X})\leq 0} f_{{\bf
           X}}({\bf x}) d {\bf x}

assuming that all random variables :math:`{\bf X}` are continuous. However,
there are three major issues related to the Equation :eq:`eq:2_70`, proposed
by [Baker2010]_:

   1. There is not always enough information to define the complete joint
      probability density function :math:`f_X({\bf x})`.
   2. The limit state function :math:`g({\bf X})` may be difficult to evaluate.
   3. Even if :math:`f_X({\bf x})` and :math:`g({\bf X})` are known, numerical
      computing of high dimensional integrals is difficult.

For this reason various methods have been developed to overcome these chal-
lenges. The most common ones are the Monte Carlo simulation method and the
First Order Reliability Method (FORM).

The Classical Approach
----------------------

Before discussing more general methods, the principles are shown on a
"historical" and simplified limit state function.

.. math::
   :label: eq:2_71

           g(R,S) = R - S

Where :math:`R` is a random variable for the resistance with the outcome
:math:`r` and :math:`S` represents a random variable for the internal strength
or stress with the outcome of :math:`s`. [Lemaire2010]_ The probability of
failure is according to Equation :eq:`eq:2_70`:

.. math::
   :label: eq:2_72

           p_f = P(R-S \leq 0) = \iint\limits_{r\leq s} f_{R,S}(r,s) d r d s

If :math:`R` and :math:`S` are independent the Equation :eq:`eq:2_72` can be
rewritten as a convolution integral, where the probability of failure
:math:`p_f` can be (numerical) computed. [Schneider2007]_

.. math::
   :label: eq:2_73

             p_f = P(R-S \leq 0) = \int_{-\infty}^{\infty} F_R(x) f_{S}(x) d x

.. figure:: images/f-02-07-a.*
   :alt: Classical Approach R − S.
   :align: center
   :scale: 50

If :math:`R` and :math:`S` are independent and :math:`R \sim N (\mu_R ,
\sigma_R )` as well as :math:`S \sim N (\mu_S , \sigma_S )` are
normally distributed, the convolution integral :eq:`eq:2_73` can be evaluated
analytically.

.. math::
   :label: eq:2_74

           M = R - S

where :math:`M` is the safety margin and also normal distributed :math:`M \sim N
(\mu_M , \sigma_M )` with the parameters

.. math::
   :label: eq:2_75

           \mu_M = \mu_R-\mu_S

.. math::
  :label: eq:2_76

          \sigma_M = \sqrt{\sigma_R^2+\sigma_S^2}

The probability of failure :math:`p_f` can be determined by the use of the
standard normal distribution function.

.. math::
   :label: eq:2_77

           p_f = \Phi\left(\frac{0-\mu_m}{\sigma_M}\right)=\Phi(-\beta)

Where :math:`\beta` is the so called Cornell reliability index, named after
Cornell (1969), and is equal to the number of the standard derivation
:math:`\sigma_M` by which the mean values :math:`\mu_M` of the safety margin
:math:`M` are zero. [Faber2009]_

.. figure:: images/f-02-08-a.*
   :alt: Safety Margin an Reliability Index
   :align: center
   :scale: 50


Hasofer and Lind Reliability Index
----------------------------------

The reliability index can be interpreted as a measure of the distance to the
failure surface, as shown in the Figure above. In the one dimensional case the
standard deviation of the safety margin was used as scale. To obtain a similar
scale in the case of more basic variables, Hasofer and Lind (1974) proposed a
non-homogeneous linear mapping of a set of random variables :math:`{\bf X}`
from a physical space into a set of normalized and uncorrelated random
variables :math:`{\bf Z}` in a normalized space. [Madsen2006]_

.. note::
   [Hasofer and Lind Reliability Index]

   The Hasofer and Lind reliability index, denoted by :math:`\beta_{HL}`, is
   the shortest distance :math:`{\bf z}^*` from the origin to the failure
   surface :math:`g({\bf Z})` in a normalized space.

   .. math::
      :label: eq:2_78

              \beta_{HL}:=\beta={\vec\alpha}^T{\bf z}^*

The shortest distance to the failure surface :math:`{\bf z}^*` is also known
as design point and :math:`{\vec \alpha}` denotes the normal vector to the
failure surface :math:`g({\bf Z})` and is given by

.. math::
   :label: eq:2_79

           {\vec\alpha} = - \frac{\nabla g({\bf z}^*)}{|\nabla g({\bf z}^*)|}

where :math:`g({\bf z})` is the gradient vector, which is assumed to exist:
[Madsen2006]_

.. math::
   :label: eq:2_80

           \nabla g({\bf z}) = \left (\frac{\partial g}{\partial z_1}({\bf
           z}),\ldots, \frac{\partial g}{\partial z_n}({\bf z})\right)

Finding the reliability index :math:`\beta` is therefore an optimization
problem

.. math::
   :label: eq:2_81

           \min_x \, |{\bf z}|\,: \, g({\bf z})=0

The calculation of :math:`\beta` can be undertaken in a number of different
ways. In the general case where the failure surface is non-linear, an
iterative method must be used. [Thoft-Christensen]_

Probability Transformation
==========================

Due to the reliability index :math:`\beta_{HL}` , being only defined in a
normalized space, the basic random variables :math:`\bf X` have to be
transformed into standard normal random variables :math:`\bf Z`. Additionally,
the basic random variables :math:`\bf Z` can be correlated and those
relationships should also be transformed. 

Transformation of Dependent Random Variables using Nataf Approach
-----------------------------------------------------------------

One method to handle this is using the Nataf joint distribution model, if the
marginal cdfs are known. [Baker2010]_ The correlated random variables
:math:`{\bf X} = ( X_1 , \dots , X_n )` with the correlation matrix :math:`\bf
R` can be transformed by

.. math::
   :label: eq:2_82

           y_i=\Phi^{-1}\left(F_{X_{i}}(x_i)\right) \qquad i = 1,\dots,n

into normally distributed random variables :math:`\bf Y` with zero means and
unit variance, but still correlated with :math:`{\bf R}_0` . Nataf’s
distribution for :math:`\bf X` is obtained by assuming that :math:`\bf Y` is
jointly normal. [Liu1986]_

The correlation coefficients for :math:`\bf X` and :math:`\bf Y` are related by

.. math::
   :label: eq:2_83

           \rho_{X_i,X_j} =
           \int\limits_{-\infty}^{\infty}\int\limits_{-\infty}^{\infty}
           \left(\frac{x_i-\mu_{X_i}}{\sigma_{X_i}}\right)
           \left(\frac{x_j-\mu_{X_j}}{\sigma_{X_j}}\right)
           \frac{1}{2\pi \sqrt{1-\rho_{Y_i,Y_j}^2}}
           \exp\left(-\frac{y_i^2-2\rho_{Y_i,Y_j}y_iy_j+y_j^2}{2(1-\rho_{Y_i,Y_j}^2)}\right) d z_i d z_j

Once this is done, the transformation from the correlated normal random
variables :math:`\bf Y` to uncorrelated normal random variables :math:`\bf Z`
is addressed. Hence, the transformation is

.. math::
   :label: eq:2_84

           {\bf z}={\bf L}_0^{-1}{\bf y} \quad \Leftrightarrow \quad {\bf y} =
           {\bf L}_0{\bf z}

where :math:`\bf L` is the Cholesky decomposition of the correlation matrix
:math:`\bf R` of :math:`\bf Y`. The Jacobian matrix, denoted by :math:`\bf J`,
for the transformation is given by

.. math::
   :label: eq:2_85

           {\bf J}_{ZX} = \frac{\partial {\bf z}}{\partial {\bf x}} = {\bf
           L}_0^{-1}\text{diag} \left(\frac{f_{X_i}(x_i)}{\Phi (z_i)}\right)

This approach is useful when the marginal distribution for the random
variables :math:`\bf X` is known and the knowledge about the variables
dependence is limited to correlation coefficients. [Baker2010]_
[DerKiureghian2006]_

Transformation of Dependent Random Variables using Rosenblatt Approach
----------------------------------------------------------------------

An alternative to the Nataf approach is to consider the joint pdf of
:math:`\bf X` as a product of conditional pdfs.

.. math::
   :label: eq:2_86

           f_{{\bf X}}({\bf x}) = f_{X_1}(x_1) f_{X_2|X_1}(x_2|x_1) \dots
           f_{X_n|X_1,\dots,X_{n-1}}(x_n|x_1,\dots,x_{n-1})

As a result of the sequential conditioning in the pdf, the conditional cdfs
are given for :math:`i \in [1,n]`

.. math::
   :label: eq:2_87

           F_{X_i|X_1,\dots,X_{i-1}}(x_i|x_1,\dots,x_{i-1}) =
           \int_{-\infty}^{x_i}
           f_{X_i|X_1,\dots,X_{i-1}}(x_i|x_1,\dots,x_{i-1}) d x_i

These conditional distributions for the random variables :math:`\bf X` can be
transformed into standard normal marginal distributions for the variables
:math:`\bf Z`, using the so called Rosenblatt transformation
[Rosenblatt1952]_, suggested by Hohenbichler and Rackwitz (1981).

.. math::
   :label: eq:2_88

           \begin{split}
           z_1 &= \Phi^{-1}\left( F_{X_1}(x_1) \right)\\
           z_2 &= \Phi^{-1}\left( F_{X_2|X_1}(x_2|x_1) \right)\\
           &\vdots\\
           z_n &= \Phi^{-1}\left(
           F_{X_n|X_1,\dots,X_{n-1}}(x_n|x_1,\dots,x_{n-1}) \right)
           \end{split}

The Jacobian of this transformation is a lower triangular matrix having the
elements [Baker2010]

.. math::
   :label: eq:2_89

           \left[{\bf J}_{ZX}\right]_{i,j} = \frac{\partial z_i}{\partial x_j} = 
           \begin{cases}\displaystyle
           \frac{1}{\Phi(u_i)}\frac{\partial}{\partial x_j}
           F_{X_i|X_1,\dots,X_{i-1}}(x_i|x_1,\dots,x_{i-1}) & i \geq j\\
           0 & i < j
           \end{cases}

In some cases the Rosenblatt transformation cannot be applied, because the
required conditional pdfs cannot be provided. In this case other
transformations may be useful, for example Nataf transformation.
[Faber2009]_

First-Order Reliability Method (FORM)
=====================================

Let :math:`\bf Z` be a set of uncorrelated and standardized normally distributed random
variables :math:`( Z_1 ,\dots, Z_n )` in the normalized z-space, corresponding
to any set of random variables :math:`{\bf X} = ( X_1 , \dots , X_n )` in the
physical x-space, then the limit state surface in x-space is also mapped on
the corresponding limit state surface in z-space.

According to Definition :eq:`eq:2_78`, the reliability index :math:`\beta` is
the minimum distance from the z-origin to the failure surface. This distance
:math:`\beta` can directly be mapped to a probability of failure

.. math::
   :label: eq:2_90

           p_f \approx p_{f1} = \Phi(-\beta)

this corresponds to a linearization of the failure surface. The linearization point
is the design point :math:`{\bf z}^*`. This procedure is called First Order
Reliability Method (FORM) and :math:`\beta` is the First Order Reliability
Index. [Madsen2006]_


.. figure:: images/f-02-09-a.*
   :alt: FORM a
   :align: center
   :scale: 50

Representation of a physical space with a set :math:`{\bf X}` of any two
random variables. The shaded area denotes the failure domain and
:math:`g({\bf X}) = 0` the failure surface.

.. figure:: images/f-02-09-b.*
   :alt: FORM b
   :align: center
   :scale: 50

After
transformation in the normalized space, the random variables :math:`{\bf X}`
are now uncorrelated and standardized normally distributed, also the failure
surface is transformed into :math:`g({\bf Z}) = 0`.

.. figure:: images/f-02-09-c.*
   :alt: FORM c
   :align: center
   :scale: 50

FORM corresponds to a linearization of the failure surface :math:`g({\bf Z}) =
0`. Performing this method, the design point :math:`{\bf z}^*` and the
reliability index :math:`\beta` can be computed.


Second-Order Reliability Method (SORM)
======================================

FORM approximates the failure surface :math:`g({\bf Z}) = 0` by a tangent
hyperplane at the design point.  When the failure surface has significant
curvature at the design point, this linear approximation can over- or
under-estimate :math:`p_f`.  The Second-Order Reliability Method (SORM)
improves on FORM by fitting a quadratic surface (paraboloid) to
:math:`g({\bf Z}) = 0` at the design point, thereby capturing
second-order effects [Baker2010]_.

Quadratic approximation in rotated space
-----------------------------------------

Starting from the FORM design point :math:`{\bf z}^*` and the unit
direction vector :math:`\boldsymbol{\alpha} = -{\bf z}^*/\beta`, the
standard normal space is rotated so that :math:`{\bf z}^*` lies at
distance :math:`\beta` along the last axis.  Let :math:`{\bf R}` denote
the orthonormal rotation matrix constructed by Gram--Schmidt
orthonormalisation with :math:`\boldsymbol{\alpha}` in the last row, and
let :math:`{\bf u}' = {\bf R}\,{\bf z}` be coordinates in the rotated
space.  In these coordinates the failure surface is approximated as:

.. math::

    g({\bf u}') \approx \beta - u'_n
    + \tfrac{1}{2} \sum_{i=1}^{n-1} \kappa_i \,(u'_i)^2

where :math:`\kappa_i` are the *principal curvatures* of the failure
surface at the design point and :math:`u'_n` is the coordinate along the
design-point direction.  Positive curvature means the failure surface
curves away from the origin (conservative with respect to FORM); negative
curvature means it curves towards the origin (unconservative).

The key task is to determine the principal curvatures :math:`\kappa_i`.
Pystra provides two approaches.

Curve-Fitting
-------------

The default method (``fit_type='cf'``) obtains the curvatures from the
Hessian matrix of the limit state function.  The Hessian :math:`{\bf H}`
of :math:`g` at the design point :math:`{\bf z}^*` is computed by finite
differences of the gradient that is already available from FORM.  This
matrix is then rotated and normalised:

.. math::

    {\bf A} = \frac{{\bf R}\,{\bf H}\,{\bf R}^T}
    {\lVert \nabla g({\bf z}^*) \rVert}

The principal curvatures :math:`\kappa_i` are the eigenvalues of the
leading :math:`(n{-}1) \times (n{-}1)` sub-matrix of :math:`{\bf A}`
(i.e.\ the block excluding the last row and column, which corresponds to
the design-point direction).  These curvatures are symmetric: the
paraboloid has the same curvature on both sides of each principal axis.

The Breitung approximation [Breitung1984]_ then gives the second-order
failure probability:

.. math::
    :label: eq:sorm_breitung

    p_{f2} = \Phi(-\beta) \prod_{i=1}^{n-1}
    \left(1 + \kappa_i \,\beta\right)^{-1/2}

This result is asymptotically exact as :math:`\beta \to \infty`.

Point-Fitting
-------------

An alternative method (``fit_type='pf'``) determines the curvatures by
locating fitting points directly on the failure surface, without computing
the Hessian.  For each of the :math:`n{-}1` principal axes in the rotated
space, a pair of trial points is placed at :math:`u'_i = \pm k\beta`
(where :math:`k` is an adaptive step coefficient), with all other
off-axis coordinates set to zero and :math:`u'_n = \beta`.  Newton
iteration along the :math:`u'_n`-direction then drives each point onto
the surface :math:`g = 0`.

Once a fitting point has converged, its curvature is computed from the
displacement along the design-point direction:

.. math::

    \kappa_i = \frac{2\,(u'_n - \beta)}{(u'_i)^2}

Because points are fitted on both the positive and negative sides of each
axis, the method yields asymmetric curvatures :math:`\kappa_i^+` and
:math:`\kappa_i^-`.  The generalised Breitung formula for asymmetric
curvatures is:

.. math::
    :label: eq:sorm_breitung_pf

    p_{f2} = \Phi(-\beta) \prod_{i=1}^{n-1} \frac{1}{2}
    \left[ \left(1 + \beta\, \kappa_i^+\right)^{-1/2}
         + \left(1 + \beta\, \kappa_i^-\right)^{-1/2} \right]

When the curvatures are symmetric (:math:`\kappa_i^+ = \kappa_i^-`), this
reduces to the standard Breitung formula :eq:`eq:sorm_breitung`.

Hohenbichler--Rackwitz Modification
------------------------------------

The Breitung formula is asymptotically exact for large :math:`\beta` but
can be inaccurate for moderate values.  Hohenbichler and Rackwitz
[Hohenbichler1988]_ proposed replacing :math:`\beta` in the curvature
terms with:

.. math::

    \psi = \frac{\phi(\beta)}{\Phi(-\beta)}

where :math:`\phi` is the standard normal PDF.  The quantity :math:`\psi`
is the conditional mean of the standard normal distribution given that it
exceeds :math:`\beta`, and provides a better local expansion for moderate
reliability indices.  The modified formula is:

.. math::
    :label: eq:sorm_hr

    p_{f2}^{\text{HR}} = \Phi(-\beta) \prod_{i=1}^{n-1}
    \left(1 + \psi\, \kappa_i\right)^{-1/2}

with the obvious extension to asymmetric curvatures from point-fitting.
Both the standard and modified Breitung results are reported by Pystra.

Validity and method comparison
-------------------------------

The Breitung and Hohenbichler--Rackwitz formulas require each curvature
term in the product to be positive.  For the standard Breitung formula
this means :math:`\kappa_i > -1/\beta`; for the modified formula,
:math:`\kappa_i > -1/\psi`.  If any curvature violates this bound the
approximating paraboloid opens towards the origin and the second-order
approximation is invalid.

The two fitting methods offer different trade-offs:

- **Curve-fitting** requires fewer limit state evaluations (one gradient
  perturbation per random variable) and produces symmetric curvatures.  It
  is well suited to smooth failure surfaces where the curvature is
  approximately the same on both sides of the design point.

- **Point-fitting** requires more evaluations (Newton iteration for each
  of :math:`2(n{-}1)` fitting points) but captures asymmetric curvature.
  This is advantageous when the failure surface has markedly different
  shapes on each side of the design point, as can occur with non-linear
  limit state functions.


System Reliability
==================

System reliability concerns a structure whose failure is governed by more than
one component event.  If the component limit states are
:math:`g_i({\bf X})`, the component failure events are

.. math::

   F_i = \{g_i({\bf X}) \leq 0\}.

For a series system the system failure event is the union of component failure
events,

.. math::

   F_\mathrm{series} = \bigcup_{i=1}^{n} F_i,

and the equivalent scalar limit-state function can be written as

.. math::

   g_\mathrm{series}({\bf X}) = \min_i g_i({\bf X}).

For a parallel system the system failure event is the intersection of component
failure events,

.. math::

   F_\mathrm{parallel} = \bigcap_{i=1}^{n} F_i,

with equivalent scalar limit-state function

.. math::

   g_\mathrm{parallel}({\bf X}) = \max_i g_i({\bf X}).

These min/max forms are useful because they preserve the standard Pystra sign
convention: positive means safe and non-positive means failed.  They also
allow the same system definition to be passed to simulation methods, active
learning, and, when the envelope is sufficiently smooth near the controlling
point, FORM/SORM.

k-of-n, Cut-Set, and Tie-Set Systems
------------------------------------

More general topologies are often described in terms of events rather than a
single analytic limit-state expression [Ditlevsen2007]_.  A k-of-n system
fails when at least :math:`k` component events have occurred:

.. math::

   F_{k|n} =
   \left\{\sum_{i=1}^{n} I(F_i) \geq k\right\},

where :math:`I(F_i)` is one if event :math:`F_i` occurs and zero otherwise.
This representation is exact for Boolean enumeration and simulation, but it is
not generally differentiable.

If the minimum cut sets :math:`C_m` are known, the system failure event can be
written as

.. math::

   F_\mathrm{sys} =
   \bigcup_{m=1}^{n_c} \left(\bigcap_{i \in C_m} F_i\right).

The dual path, or tie-set, representation writes the safe event as the union
of working tie sets.  For tie sets :math:`T_m`,

.. math::

   S_\mathrm{sys} =
   \bigcup_{m=1}^{n_t} \left(\bigcap_{i \in T_m} \bar{F}_i\right).

Cut-set and tie-set descriptions are common in structural system reliability
because they let the engineer encode known collapse mechanisms or load paths
without enumerating every possible Boolean state [Song2003]_.

Ditlevsen Bounds
----------------

For a series system, exact evaluation of
:math:`P(\cup_i F_i)` may require high-dimensional integration over a union of
failure domains.  If the component probabilities :math:`P(F_i)` and pairwise
intersections :math:`P(F_i \cap F_j)` are available, Ditlevsen's bounds give a
second-order estimate of the union probability [Ditlevsen1979]_.  For a chosen
event ordering, the lower bound is

.. math::

   P_L =
   P(F_1) +
   \sum_{i=2}^{n}
   \max\left[
      P(F_i) - \sum_{j=1}^{i-1} P(F_i \cap F_j),\ 0
   \right],

and the upper bound is

.. math::

   P_U =
   \sum_{i=1}^{n} P(F_i)
   - \sum_{i=2}^{n} \max_{1 \leq j < i} P(F_i \cap F_j).

The bounds depend on event ordering.  For small systems the ordering can be
checked exhaustively; for large systems, the ordering should be chosen using
engineering judgement or a heuristic.  Mainçon's 100-element series-system
benchmark is a useful validation case because it reports component and
pairwise probabilities directly [Maincon2000]_.

Linear-programming bounds generalise this idea to arbitrary systems and
arbitrary available event information.  Song and Der Kiureghian showed that LP
bounds can use component, pairwise, and higher-order event probabilities for
general cut-set systems, including the rigid-plastic cantilever-bar benchmark
[Song2003]_.  This is a natural extension beyond the current Ditlevsen bounds
API.

Four-Branch Case
----------------

The four-branch case is a widely used benchmark for reliability algorithms
because it has multiple disconnected failure regions [Schueremans2005]_.  With
independent standard normal variables :math:`X_1` and :math:`X_2`, it is
defined by

.. math::

   g_\mathrm{FBC}({\bf X}) = \min(g_1, g_2, g_3, g_4),

where

.. math::

   \begin{aligned}
   g_1 &= 3 + 0.1(X_1 - X_2)^2 - \frac{X_1 + X_2}{\sqrt{2}}, \\
   g_2 &= 3 + 0.1(X_1 - X_2)^2 + \frac{X_1 + X_2}{\sqrt{2}}, \\
   g_3 &= (X_1 - X_2) + \frac{6}{\sqrt{2}}, \\
   g_4 &= (X_2 - X_1) + \frac{6}{\sqrt{2}}.
   \end{aligned}

The benchmark is a series system in event terms, but a single FORM analysis
can find only one local design point.  Simulation, subset simulation, and
active-learning methods are therefore better suited to estimating the global
failure probability unless a dedicated first-order system reliability method
is used.

First-Order System Reliability
------------------------------

First-order system reliability methods approximate each component failure
surface near its design point and then integrate the resulting system event in
standard normal space.  This requires more information than a scalar topology:
component design points, component normal vectors, dependence between
linearised events, and a clear isoprobabilistic transformation.  Rosenblatt
transformations add an additional ordering issue because the transformed
standard-space geometry can depend on the conditioning order [Meinen2025]_.

For this reason Pystra currently separates three tasks:

1. users encode the system topology using series, parallel, k-of-n, cut-set,
   or tie-set systems;
2. existing simulation and active-learning methods estimate the resulting
   failure probability directly;
3. analytical bounds such as Ditlevsen bounds are computed from event
   probabilities when those probabilities are available.


Simulation Methods
==================

The preceding sections describe some methods for determining the reliability
index :math:`\beta` for some common forms of the limit state
function. However, it is sometimes extremely difficult or impossible to find
:math:`\beta`. [Nowak2000]_

In this case, Equation :eq:`eq:2_70` may also be
estimated by numerical simulation methods. A large variety of simulation
techniques can be found in the literature, indeed, the most commonly used
method is the Monte Carlo method. [Faber2009]_

The principle of simulation methods is to carry out random sampling in the
physical (or standardized) space. For each of the samples the limit state
function is evaluated to figure out, whether the configuration is desired or
undesired. The probability of failure :math:`p_f` is estimated by the number
of undesired configurations, respected to the total numbers of
samples. [Lemaire2010]_

For this analysis Equation :eq:`eq:2_70` can be rewritten as

.. math::
   :label: eq:2_91

           p_f = P(g({\bf X})\leq 0) = \iiint\limits_{g({\bf X})\leq 0}
           I(g({\bf X})\leq 0) f_{{\bf X}}({\bf x}) d {\bf x}

where :math:`I` is an indicator function that is equals to 1 if :math:`g({\bf
X}) \leq 0` and otherwise 0. Equation :eq:`eq:2_91` can be interpreted as
expected value of the indicator function. Therefore, the probability of
failure can be estimated such as [Malioka2009]_

.. math::
   :label: eq:2_92

           \tilde{p}_f = \text{Ex}\left[I(g({\bf X})\leq 0)\right] =
           \frac{1}{n}\sum_{i=1}^{n} I(g({\bf X})\leq 0)

Crude Monte Carlo Simulation
============================

The Crude Monte Carlo simulation (CMC) is the most simple form and corresponds
to a direct application of Equation :eq:`eq:2_92`. A large number :math:`n` of
samples are simulated for the set of random variables :math:`\bf X`. All
samples that lead to a failure are counted :math:`n_f` and after all
simulations the probability of failure :math:`p_f` may be estimated by
[Faber2009]_

.. math::
   :label: eq:2_93

           \tilde{p}_f = \frac{n_f}{n}

Theoretically, an infinite number of simulations will provide an exact
probability of failure. However, time and the power of computers are limited;
therefore, a suitable amount of simulations :math:`n` are required to achieve
an acceptable level of accuracy. One possibility to reach such a level is to
limit the coefficient of variation CoV for the probability of
failure. [Lemaire2010]_

.. math::
   :label: eq:2_94

           \text{CoV} = \sqrt{\frac{1-p_f}{n p_f}} \approx \frac{1}{\sqrt{n
           p_f}} \qquad \text{for} \quad p_f \to 0

Importance Sampling
===================

To decrease the number of simulations and the coefficient of variation, other
methods can be performed. One commonly applied method is the Importance
Sampling simulation method (IS). Here the prior information about the failure
surface is added to Equation :eq:`eq:2_91`

.. math::
   :label: eq:2_95

           p_f = P(g({\bf X})\leq 0) = \iiint\limits_{g({\bf X})\leq 0}
           I(g({\bf X})\leq 0) \frac{f_{{\bf X}}({\bf x})}{h_{{\bf X}}({\bf
           x})} h_{{\bf X}}({\bf x}) d {\bf x}

where :math:`h_{X} ({\bf X})` is the importance sampling probability
density function of :math:`\bf X`. Consequently Equation :eq:`eq:2_92` is
extended to [Faber2009]_

.. math::
  :label: eq:2_96

          \tilde{p}_f = \text{Ex}\left[I(g({\bf X})\leq 0) \frac{f_{{\bf
          X}}({\bf x})}{h_{{\bf X}}({\bf x})}\right] =
          \frac{1}{n}\sum_{i=1}^{n} I(g({\bf X})\leq 0)\frac{f_{{\bf X}}({\bf
          x})}{h_{{\bf X}}({\bf x})}

The key to this approach is to choose :math:`h_{X} ({\bf X})` so that samples
are obtained more frequently from the failure domain. For this reason, often a
FORM (or SORM) analysis is performed to find a prior design point. [Baker2010]

.. figure:: images/f-02-10-a.*
   :alt: MC a
   :align: center
   :scale: 50

Representation of a physical space with a set :math:`{\bf X}` of any two
random variables. The shaded area denotes the failure domain and `g({\bf X}) =
0` the failure surface.

.. figure:: images/f-02-10-b.*
   :alt: MC b
   :align: center
   :scale: 50

For the CMC method every dot corresponds to one configuration of the random
variables :math:`{\bf X}`. Dots in shaded areas lead to a failure.

.. figure:: images/f-02-10-c.*
   :alt: MC c
   :align: center
   :scale: 50


The IS simulation method uses a distribution centered on the design point
:math:`{\bf x}^*`, is obtained from a FORM (or SORM) analysis. More dots in
the failure domain can be observed.


Line Sampling
=============

Line Sampling (LS) is a variance-reduction technique that exploits the
important direction :math:`\boldsymbol{\alpha}` identified by FORM to reduce
the n-dimensional sampling problem to a family of one-dimensional problems
[Koutsourelakis2004]_.

The important direction :math:`\boldsymbol{\alpha}` is the unit vector from
the origin in standard-normal space toward the most probable failure point.
For each of :math:`N` random samples :math:`\mathbf{u}_i` drawn from
:math:`\mathcal{N}(\mathbf{0}, \mathbf{I})`, the component along
:math:`\boldsymbol{\alpha}` is projected out to obtain the foot-point

.. math::

   \mathbf{v}_i = \mathbf{u}_i
       - \left(\mathbf{u}_i^T \boldsymbol{\alpha}\right) \boldsymbol{\alpha}

which lies in the :math:`(n-1)`-dimensional hyperplane perpendicular to
:math:`\boldsymbol{\alpha}`. A root-finding step then locates the scalar
:math:`c_i` such that

.. math::

   g\!\left(\mathbf{v}_i + c_i\,\boldsymbol{\alpha}\right) = 0

The failure probability is estimated as the average of the one-dimensional
conditional failure probabilities along each line:

.. math::
   :label: eq_ls_pf

   \hat{p}_f = \frac{1}{N} \sum_{i=1}^{N} \Phi(-c_i)

where :math:`\Phi` is the standard normal CDF.  Each term
:math:`\Phi(-c_i)` is the probability that a point drawn from
:math:`\mathcal{N}(0,1)` along the :math:`i`-th line lies in the failure
domain.

The variance of the estimator is

.. math::

   \widehat{\operatorname{Var}}\!\left[\hat{p}_f\right]
       = \frac{1}{N}\,\operatorname{Var}\!\left[\Phi(-c_i)\right]

giving a coefficient of variation

.. math::

   \text{CoV} = \frac{\operatorname{Std}\!\left[\Phi(-c_i)\right]}{\sqrt{N}\,\hat{p}_f}

Line Sampling is particularly efficient when the failure surface is
nearly planar near the design point, because all :math:`c_i` are then
close to :math:`\beta_{\text{FORM}}` and
:math:`\operatorname{Var}[\Phi(-c_i)]` is small.


Subset Simulation
=================

Subset Simulation (SS) is an adaptive simulation method that decomposes the
rare failure event :math:`F = \{g(\mathbf{u}) \le 0\}` into a sequence of
more frequent nested intermediate events [AuBeck2001]_:

.. math::

   F_1 \supset F_2 \supset \cdots \supset F_m = F

where :math:`F_j = \{g(\mathbf{u}) \le y_j\}` and the thresholds satisfy
:math:`y_1 > y_2 > \cdots > y_m = 0`.  By the chain rule of probability,

.. math::
   :label: eq_ss_pf

   p_f = P(F_1) \prod_{j=2}^{m} P(F_j \mid F_{j-1})

Each conditional probability is targeted at a user-specified level
:math:`p_0` (typically 0.1), making every factor in the product relatively
large and easy to estimate.

**Algorithm**

1. **Level 0** — Generate :math:`N` samples from
   :math:`\mathcal{N}(\mathbf{0}, \mathbf{I})` and evaluate the LSF.
   Choose :math:`y_1` as the :math:`p_0`-th quantile of the LSF values,
   so that :math:`N p_0` samples satisfy :math:`g \le y_1`.  If
   :math:`y_1 \le 0`, the failure probability is estimated directly as
   :math:`\hat{p}_f = N_{\text{fail}} / N`.

2. **Levels** :math:`j \ge 1` — Use the :math:`N p_0` samples satisfying
   :math:`g \le y_{j-1}` as seeds for Modified Metropolis--Hastings (MMH)
   chains.  Generate :math:`N` new samples distributed approximately as
   :math:`\mathcal{N}(\mathbf{0}, \mathbf{I})` conditioned on
   :math:`g \le y_{j-1}`.  Set :math:`y_j` as the :math:`p_0`-th quantile
   of the new LSF values.  Stop when :math:`y_j \le 0`.

3. **Final level** — Count the actual failures (:math:`g \le 0`) in the last
   conditional sample: :math:`\hat{p}_m = N_{\text{fail}} / N`.

4. **Estimate** — :math:`\hat{p}_f = \hat{p}_1 \hat{p}_2 \cdots \hat{p}_m`

**Modified Metropolis--Hastings (MMH)**

To generate samples from :math:`\mathcal{N}(\mathbf{0}, \mathbf{I})`
conditioned on :math:`g(\mathbf{u}) \le y_j`, the MMH algorithm applies
Metropolis updates component-wise.  For each component :math:`d`:

.. math::

   \xi_d \sim u_d + \sigma\, \mathcal{U}(-1, 1)

with acceptance probability

.. math::

   \alpha_d = \min\!\left(1,\; e^{-(\xi_d^2 - u_d^2)/2}\right)

After assembling all accepted components into a candidate
:math:`\mathbf{u}'`, the entire vector is accepted only if
:math:`g(\mathbf{u}') \le y_j`; otherwise the current state is retained.
This ensures the stationary distribution is
:math:`\mathcal{N}(\mathbf{0}, \mathbf{I}) \mid g(\mathbf{u}) \le y_j`.

**Coefficient of variation**

Ignoring correlations within the Markov chains (a lower bound on the true
variance), the CoV of the estimator is approximated by [AuBeck2001]_

.. math::

   \delta^2(\hat{p}_f) \approx \sum_{j=1}^{m} \frac{1 - \hat{p}_j}{N\,\hat{p}_j}

Subset Simulation is particularly effective for small failure probabilities
(roughly :math:`p_f < 10^{-3}`), where crude Monte Carlo would require an
impractically large number of samples.  A benchmark comparison of simulation
methods on high-dimensional problems is given in [Schueller2007]_.


Active Learning Reliability
===========================

When the limit state function :math:`g(\mathbf{X})` is expensive to
evaluate — for instance because each call involves a finite-element
analysis — the simulation methods described above may be impractical.
Active Learning Reliability (ALR) addresses this by constructing a cheap
surrogate model of :math:`g` and using it in place of the true function
for the Monte Carlo classification, while adaptively selecting new
training points to improve the surrogate precisely where it matters: near
the failure surface :math:`g = 0` [Echard2011]_ [Moustapha2022]_.

The general framework is known as **AK-MCS** (Active Kriging — Monte
Carlo Simulation) when Kriging is the surrogate, though the algorithm
is surrogate-agnostic.

Algorithm
---------

1. **Initial experimental design** — Generate :math:`N_0 =
   \max(10, 2n)` points in standard-normal space by Latin Hypercube
   Sampling (LHS), transform to physical space, and evaluate the true
   limit state function.

2. **Candidate population** — Draw a large Monte Carlo sample
   :math:`\mathcal{S}` of size :math:`N_{\text{cand}}` (typically
   :math:`10^4`) from the joint input distribution.  This population is
   *never* evaluated on the true LSF; it is only used with the surrogate.

3. **Iterative enrichment loop:**

   a. Fit the surrogate to the current experimental design
      :math:`(\mathbf{X}_{\text{ED}}, \mathbf{g}_{\text{ED}})`.

   b. Predict :math:`\hat{\mu}(\mathbf{x})` and
      :math:`\hat{\sigma}(\mathbf{x})` for every candidate
      :math:`\mathbf{x} \in \mathcal{S}`.

   c. Estimate the failure probability from the surrogate:
      :math:`\hat{p}_f = N_{\text{fail}} / N_{\text{cand}}` where
      :math:`N_{\text{fail}} = \#\{\mathbf{x} \in \mathcal{S} :
      \hat{\mu}(\mathbf{x}) \le 0\}`.

   d. Evaluate the **learning function** on the candidates and select
      the most informative point :math:`\mathbf{x}^*`.

   e. Evaluate the **true** LSF at :math:`\mathbf{x}^*` and add it to
      the experimental design.

   f. Check **convergence**; if not met, return to step (a).

4. **Final estimate** — Refit the surrogate on the final design and
   classify the candidate population to obtain :math:`\hat{p}_f` and
   :math:`\hat{\beta} = -\Phi^{-1}(\hat{p}_f)`.

The total number of expensive LSF evaluations is typically 30–50, making
ALR orders of magnitude cheaper than crude Monte Carlo for problems where
each evaluation takes minutes or hours.

Surrogate Models
----------------

**Kriging (Gaussian Process Regression)**

Kriging models the limit state function as a realisation of a Gaussian
process with prior mean :math:`\mu(\mathbf{x})` and covariance kernel
:math:`k(\mathbf{x}, \mathbf{x}')`.  Given training data, the posterior
at a new point :math:`\mathbf{x}` is Gaussian with predictive mean
:math:`\hat{\mu}(\mathbf{x})` and variance
:math:`\hat{\sigma}^2(\mathbf{x})`.  This natural uncertainty
quantification makes Kriging ideally suited to active learning: the
learning functions exploit :math:`\hat{\sigma}` to identify where the
surrogate is least certain about the sign of :math:`g`.

Pystra uses the Matérn 5/2 covariance kernel with automatic
hyperparameter optimisation via maximum likelihood, and standardises the
inputs to unit variance before fitting to handle variables with very
different physical scales.

**Polynomial Chaos Expansion (PCE)**

PCE approximates the limit state function as a polynomial in the input
variables:

.. math::

   g(\mathbf{X}) \approx \sum_{\boldsymbol{\alpha} \in \mathcal{A}}
   c_{\boldsymbol{\alpha}}\, \Psi_{\boldsymbol{\alpha}}(\mathbf{X})

where :math:`\Psi_{\boldsymbol{\alpha}}` are multivariate orthogonal
polynomials and :math:`c_{\boldsymbol{\alpha}}` are coefficients
determined by least-squares regression on the experimental design.  PCE
does not provide a native variance estimate at each prediction point, so
leave-one-out cross-validation error is used as a global uncertainty
measure for the learning function.

Learning Functions
------------------

The learning function scores each candidate point by how informative it
would be if added to the experimental design.  The point with the best
score is selected for the next true LSF evaluation.

**U-function** [Echard2011]_

.. math::
   :label: eq:lf_u

   U(\mathbf{x}) = \frac{|\hat{\mu}(\mathbf{x})|}
                        {\hat{\sigma}(\mathbf{x})}

Points with small :math:`U` are close to the failure surface relative
to the prediction uncertainty — the surrogate is least confident about
their classification.  The enrichment selects
:math:`\mathbf{x}^* = \arg\min U(\mathbf{x})`.  Convergence is declared
when :math:`\min U \ge 2`, meaning every candidate is at least 2
standard deviations from the predicted limit state.

**Expected Feasibility Function (EFF)** [Bichon2008]_

.. math::
   :label: eq:lf_eff

   \text{EFF}(\mathbf{x})
   = \hat{\mu} \left[2\Phi\!\left(\frac{-|\hat{\mu}|}{\hat{\sigma}}\right)
     - \Phi\!\left(\frac{-\varepsilon - \hat{\mu}}{\hat{\sigma}}\right)
     - \Phi\!\left(\frac{-\varepsilon + \hat{\mu}}{\hat{\sigma}}\right)\right]
   + \hat{\sigma} \left[2\varphi\!\left(\frac{-|\hat{\mu}|}{\hat{\sigma}}\right)
     - \varphi\!\left(\frac{-\varepsilon - \hat{\mu}}{\hat{\sigma}}\right)
     - \varphi\!\left(\frac{-\varepsilon + \hat{\mu}}{\hat{\sigma}}\right)\right]

where :math:`\varepsilon = 2\hat{\sigma}`, :math:`\Phi` and
:math:`\varphi` are the standard normal CDF and PDF.  EFF is the
expected improvement in the feasibility classification within an
:math:`\varepsilon`-band around the failure surface.  Enrichment selects
:math:`\mathbf{x}^* = \arg\max \text{EFF}(\mathbf{x})`; convergence is
declared when :math:`\max \text{EFF} < 10^{-3}`.

Convergence Criteria
--------------------

Pystra uses a combined convergence criterion.  Both conditions must be
satisfied for :math:`N_{\text{conv}}` consecutive iterations (default 2):

1. **Learning-function criterion** — the learning function indicates
   that the surrogate is sufficiently accurate near the failure surface
   (see thresholds above).

2. **Beta stability** — the relative change in the reliability index
   satisfies :math:`|\Delta\beta / \beta| < \varepsilon_\beta` (default
   :math:`\varepsilon_\beta = 0.01`).

This dual criterion guards against premature convergence: the learning
function checks the surrogate quality near :math:`g = 0`, while beta
stability ensures the failure probability estimate has settled.


Sensitivity Analysis
====================

In structural reliability, knowing the reliability index :math:`\beta` alone
is often insufficient. Engineers also need to understand *how sensitive*
:math:`\beta` is to the parameters of the stochastic model — the means,
standard deviations, and correlation coefficients of the random variables.
This information guides decisions about where to invest in data collection
or quality control.

Pystra computes the sensitivity
:math:`\partial\beta/\partial\theta_k` for each distribution parameter
:math:`\theta_k` using two complementary approaches.

Finite-Difference Method
------------------------

The simplest approach perturbs each parameter by a small amount
:math:`\Delta\theta_k = \delta\,\sigma_k` and re-runs FORM:

.. math::
   :label: eq:fd_sens

   \frac{\partial\beta}{\partial\theta_k}
   \approx \frac{\beta(\theta_k + \Delta\theta_k) - \beta(\theta_k)}
               {\Delta\theta_k}

This requires :math:`2n + 1` FORM runs (one baseline plus two per
parameter). The method is straightforward and distribution-agnostic, but
can be numerically unstable when the perturbation changes the Nataf
transformation significantly — particularly for correlated non-normal
variables with small sensitivities.

Closed-Form Method (Bourinet 2017)
----------------------------------

A more efficient and accurate approach post-processes the converged FORM
design point to obtain exact (up to quadrature) sensitivities from a
single FORM run. This method, due to [Bourinet2017]_ (building on the
FERUM software framework [Bourinet2009]_ [Bourinet2010]_), differentiates
the Nataf transformation chain analytically.

The sensitivity of :math:`\beta` to a marginal distribution parameter
:math:`\theta_k` decomposes into two terms:

.. math::
   :label: eq:cf_sens

   \frac{\partial\beta}{\partial\theta_k}
   = \underbrace{{\boldsymbol\alpha}^T \mathbf{L}_0^{-1}
     \frac{\partial\mathbf{z}}{\partial\theta_k}}_{\text{first term}}
   + \underbrace{{\boldsymbol\alpha}^T
     \frac{\partial\mathbf{L}_0^{-1}}{\partial\theta_k}
     \mathbf{z}}_{\text{second term}}

where :math:`\boldsymbol\alpha` is the FORM direction cosine vector,
:math:`\mathbf{L}_0` is the Cholesky factor of the modified (Nataf)
correlation matrix :math:`\mathbf{R}_0`, and :math:`\mathbf{z}` is the
correlated standard-normal design point.

The first term captures how the marginal transformation changes at the
design point; the second term accounts for changes in the correlation
structure due to the parameter perturbation. For uncorrelated normal
variables, the second term vanishes identically.

The derivative of the inverse Cholesky factor is computed from:

.. math::
   :label: eq:dinvL

   \frac{\partial\mathbf{L}_0^{-1}}{\partial\theta}
   = -\mathbf{L}_0^{-1}\,
     \frac{\partial\mathbf{L}_0}{\partial\theta}\,
     \mathbf{L}_0^{-1}

where :math:`\partial\mathbf{L}_0/\partial\theta` is obtained by
simultaneously differentiating the Cholesky decomposition algorithm.

Correlation sensitivities :math:`\partial\beta/\partial\rho_{ij}` are
also available from the closed-form method. Since the marginal
transformations do not depend on the correlation coefficients, only the
second term of Equation :eq:`eq:cf_sens` contributes.

Generalised Parameter Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond mean and standard deviation, distributions may declare additional
sensitivity parameters — for example, the shape parameter :math:`\xi` of
the GEV distribution controls the tail behaviour and can significantly
influence :math:`\beta`.

Each distribution declares its sensitivity parameters via the
:attr:`~pystra.distributions.distribution.Distribution.sensitivity_params`
property.  The base class returns ``{"mean", "std"}``; subclasses with
extra parameters (e.g. GEV shape) override this to include them.  The
sensitivity pipeline then iterates over whatever parameters each
distribution declares, so both the finite-difference and closed-form
methods generalise automatically.

For shape parameters, the partial derivatives
:math:`\partial F_X / \partial\theta` and
:math:`\partial\mu / \partial\theta`,
:math:`\partial\sigma / \partial\theta` are evaluated numerically via
central differences unless the distribution provides an analytical
override.  See the :ref:`developer guide <adding_distributions>` for
implementation details.
