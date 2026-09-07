**********************
Theoretical Background
**********************

.. contents:: Outline
   :local:
   :depth: 2

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

Classical FORM uses independent standard-normal coordinates. A probability
transformation maps the joint law of the physical variables into that space.
Generalized Nataf also permits spherical non-normal standard spaces, provided
the reliability calculation uses the corresponding probability law.

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
           \exp\left(-\frac{y_i^2-2\rho_{Y_i,Y_j}y_iy_j+y_j^2}{2(1-\rho_{Y_i,Y_j}^2)}\right) d y_i d y_j

Once this is done, the transformation from the correlated normal random
variables :math:`\bf Y` to uncorrelated normal random variables :math:`\bf Z`
is addressed. Hence, the transformation is

.. math::
   :label: eq:2_84

           {\bf z}={\bf L}_0^{-1}{\bf y} \quad \Leftrightarrow \quad {\bf y} =
           {\bf L}_0{\bf z}

where :math:`\mathbf{L}_0\mathbf{L}_0^T=\mathbf{R}_0` is the
Cholesky factorisation of the correlation matrix of :math:`\bf Y`. The Jacobian matrix, denoted by :math:`\bf J`,
for the transformation is given by

.. math::
   :label: eq:2_85

           {\bf J}_{ZX} = \frac{\partial {\bf z}}{\partial {\bf x}} = {\bf
           L}_0^{-1}\text{diag} \left(\frac{f_{X_i}(x_i)}{\varphi(y_i)}\right)

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
           \frac{1}{\varphi(z_i)}\frac{\partial}{\partial x_j}
           F_{X_i|X_1,\dots,X_{i-1}}(x_i|x_1,\dots,x_{i-1}) & i \geq j\\
           0 & i < j
           \end{cases}

Here :math:`\varphi` is the standard-normal density, and the triangular
structure follows the chosen conditioning order.

In some cases the Rosenblatt transformation cannot be applied, because the
required conditional pdfs cannot be provided. In this case other
transformations may be useful, for example Nataf transformation.
[Faber2009]_


.. _theory_copulas:
.. _/theory.rst#theory-copulas:

Copulas as joint distribution specifications
--------------------------------------------

For continuous marginals, dependence is specified by a copula :math:`C`:

.. math::

   F_{\mathbf X}(\mathbf x)=C(F_1(x_1),\ldots,F_n(x_n)),\qquad
   f_{\mathbf X}(\mathbf x)=c(F_1(x_1),\ldots,F_n(x_n))\prod_i f_i(x_i).

The copula and marginals define the joint law before a transformation is
chosen. Physical Pearson correlations alone do not determine that law.
Pystra's legacy Pearson interface assumes a Gaussian copula and calibrates
its latent correlation using :eq:`eq:2_83`. An explicit ``GaussianCopula``
already specifies the latent correlation, so that calibration is bypassed.
Changing to Student-t or Frank changes the dependence law, rather than just
changing a numerical factorisation.

For Gaussian and Student-t copulas, the pairwise Kendall rank correlations
and latent unit-diagonal shape entries satisfy

.. math::

   \tau_{ij}=\frac{2}{\pi}\arcsin R_{ij}.

The resulting full shape matrix must be positive definite. Arbitrary
physical marginals can be used with the same copula. See
:doc:`notebooks/ex_copulas` for sampling and density examples, and
:ref:`chap_copulas` for the supported specification interface.

Generalized Nataf for elliptical copulas
----------------------------------------

Following [LebrunDutfoy2009a]_, let :math:`H` be the univariate CDF of a
centred elliptical representative with unit-diagonal shape matrix
:math:`R=LL^T`. Generalized Nataf is

.. math::

   w_i=H^{-1}(F_i(x_i)),\qquad \mathbf v=L^{-1}\mathbf w,
   \qquad x_i=F_i^{-1}(H((L\mathbf v)_i)).

The transformed distribution is spherical. Its coordinates are independent
only in the Gaussian case within these Gaussian/Student-t families.
For Student-t, :math:`H=T_\nu` and :math:`\mathbf V` is spherical Student-t
with identity shape. When :math:`\nu>2`, its covariance is
:math:`\nu/(\nu-2)I`; for smaller degrees of freedom, covariance does not
exist. Identity shape therefore must not be interpreted as independence.
The Nataf Jacobian is

.. math::

   J_{VX}=L^{-1}\operatorname{diag}\left(\frac{f_i(x_i)}{h(w_i)}\right),

where :math:`h=H'`. Spherical symmetry makes every unit projection have CDF
:math:`H`. Hence a tangent failure half-space at signed geometric distance
:math:`\beta` has probability

.. math::

   P_{f,\mathrm{FORM}}=H(-\beta).

For Student-t, Pystra uses :math:`T_\nu(-\beta)`, while
``get_equivalent_beta()`` reports :math:`-\Phi^{-1}(P_f)`.
SORM, system FORM, simulation methods and the current Strong Maximum Test
require independent normal coordinates and reject spherical Student-t Nataf
space. A normal Rosenblatt mapping supports these analyses with the same
Student-t copula.

Rosenblatt conditioning and approximation invariance
----------------------------------------------------

A copula supplies the conditional CDFs required by Rosenblatt. For an order
:math:`\pi`, define :math:`p_i=F_i(x_i)` and

.. math::

   q_{\pi_k}=C_{\pi_k\mid\pi_1,\ldots,\pi_{k-1}}
       (p_{\pi_k}\mid p_{\pi_1},\ldots,p_{\pi_{k-1}}),\qquad
   u_{\pi_k}=\Phi^{-1}(q_{\pi_k}).

The :math:`q_i` are independent uniforms and the :math:`u_i` are independent
standard normals. Pystra keeps coordinate arrays in original marginal order;
``u[order[k]]`` denotes the k-th conditional innovation.

For a Student-t copula, reorder the latent vector and shape first, and write
:math:`\mathbf z=L^{-1}\mathbf w`. With one-based conditional index :math:`k`,

.. math::

   s_k^2=\frac{\nu+\sum_{j<k}z_j^2}{\nu+k-1},\qquad
   q_{\pi_k}=T_{\nu+k-1}(z_k/s_k).

The inverse constructs each :math:`z_k=s_kT_{\nu+k-1}^{-1}(q_{\pi_k})`
sequentially, then recovers the latent vector and physical marginals.
This additional conditional scaling removes the dependence that remains
after Student-t Nataf whitening.

For a Gaussian copula, Cholesky Nataf and Rosenblatt coincide for a fixed
order. Different orders are related by an orthogonal change of normal
coordinates; the optimum FORM distance and probability are invariant.
Numerical optimization can still converge to different local points.
For non-Gaussian copulas, transformations need not be related orthogonally:
exact probability remains invariant, but FORM's tangent approximation can
change with conditioning order [LebrunDutfoy2009b]_. All system components
must use the same mapping and order before their normal directions are
compared.

The bivariate Frank copula used in that paper is

.. math::

   C_\theta(p_1,p_2)=-\frac{1}{\theta}\log\left[
     1+\frac{(e^{-\theta p_1}-1)(e^{-\theta p_2}-1)}{e^{-\theta}-1}\right],

with the independence limit at :math:`\theta=0`. Its conditional CDF is
:math:`C_{2\mid1}=\partial C_\theta/\partial p_1`.
For exponential rates 1 and 3, :math:`\theta=10`, and failure event
:math:`8X_1+2X_2-1\leq0`, the tutorial reproduces FORM probabilities about
0.107 and 0.122 for the two orders, compared with direct integration of
about 0.1038. This difference is approximation error, not a different
underlying probability for each order.

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



.. _theory_strong_maximum:
.. _/theory.rst#theory-strong-maximum:

Strong Maximum Test
====================

A converged local FORM design point need not represent every important
failure region. The Strong Maximum Test [DutfoyLebrun2006]_ probes an enlarged
sphere around the origin for failure points outside the candidate's vicinity.
Pystra implements the independent standard-normal case described by
`OpenTURNS <https://openturns.github.io/openturns/latest/theory/reliability_sensitivity/strong_maximum_test.html>`_.
See :doc:`notebooks/ex_strong_maximum` for geometric examples and
:ref:`chap_strong_maximum` for the API.

Sphere geometry
----------------

Let the candidate be :math:`u^*`, with :math:`\beta=\|u^*\|>0`, and assume
that the origin is strictly safe. A density ratio :math:`0<\varepsilon<1`
defines the relevant normal-density radius :math:`r_\varepsilon`:

.. math::

   \frac{\varphi_n(r_\varepsilon e)}{\varphi_n(u^*)}=\varepsilon,
   \qquad r_\varepsilon=\sqrt{\beta^2-2\log\varepsilon},
   \qquad \|e\|=1.

With enlargement factor :math:`\tau>1`, the sampled sphere has radius

.. math::

   r=\beta+\tau(r_\varepsilon-\beta)
     =\beta(1+\tau\delta_\varepsilon),\qquad
   \delta_\varepsilon=r_\varepsilon/\beta-1.

Independent Gaussian directions normalized to length :math:`r` give uniform
sphere samples. A point is near the candidate when

.. math::

   u\cdot\frac{u^*}{\beta}>\beta,
   \quad\text{equivalently}\quad
   \cos\angle(u,u^*)>\frac{\beta}{r}.

Crossing near/far with safe/failure gives four retained point groups.
Pystra classifies failure by :math:`g<0`. Far failure points are possible
restart locations for additional design-point searches, not optimized design
points themselves. The magnitude of :math:`g` does not measure a region's
probability importance.

Cap probability and evaluation budget
--------------------------------------

The reference detection cap has half-angle
:math:`\theta=\arccos(r_\varepsilon/r)`, which differs from the candidate
vicinity angle :math:`\arccos(\beta/r)`. For dimension :math:`n>1`, its
normalized surface area is

.. math::

   p_\mathrm{cap}=\frac12 I_{\sin^2\theta}
       \left(\frac{n-1}{2},\frac12\right),

where :math:`I` is the regularized incomplete beta function. In one dimension
the sphere consists of two points and :math:`p_\mathrm{cap}=1/2`.
For :math:`N` independent samples, nominal cap-detection confidence is

.. math::

   c_N=1-(1-p_\mathrm{cap})^N,\qquad
   N=\left\lceil\frac{\log(1-c)}{\log(1-p_\mathrm{cap})}\right\rceil.

Pystra rounds upward to meet the requested nominal confidence; OpenTURNS'
reference implementation rounds to the nearest integer. Users can specify
confidence or a fixed count, with a hard ``max_points`` budget checked before
sphere evaluation. The total is :math:`N+2` point evaluations including the
origin and boundary checks. The cap can become small in high dimensions,
so inspect the budget before using an expensive structural model.

Interpretation and limitations
-------------------------------

Nominal confidence is a sampling statement about hitting a fixed cap under
the test's local-plane and failure-region extent assumptions. It is not a
posterior probability that FORM is correct, a failure-probability estimate,
or a bound on approximation error. A bounded failure island entirely inside
the sphere cannot be detected, regardless of sample count. The tutorial
constructs such an island closer to the origin than the supplied candidate.

Use the diagnostic on individual ``SystemForm.component_results`` to look
for missed regions within each component. Checking every component does not
validate the system probability. With a non-Gaussian copula, use Rosenblatt
so that the sphere geometry and cap probability apply in independent normal
space. Generalized Student-t Nataf would require different density-radius
geometry and is not supported by this implementation.

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


Load Combinations and FBC Processes
===================================

Load combination reliability problems usually distinguish permanent actions,
resistance variables, and variable actions that fluctuate in time.  The
Ferry-Borges-Castanheta (FBC) model represents a variable action as a
rectangular-wave stochastic process: the process is constant during a basic
interval :math:`\tau`, and a new independent value is drawn for each
successive interval.  This model is a standard basis for probabilistic load
combination analysis in structural reliability texts [Thoft-Christensen]_
[Madsen2006]_ [Ditlevsen2007]_ [Melchers1999]_.

If :math:`F_Q(q)` is the distribution of the action value in one basic
interval and :math:`T` is the reference period, the maximum over that period
has distribution

.. math::
   :label: eq:fbc_max_distribution

   F_{Q,\max,T}(q) = F_Q(q)^r,
   \qquad r = \frac{T}{\tau}

where :math:`r` is the number of basic intervals in the reference period.
Equivalently, when a code or statistical model supplies a maximum
distribution over duration :math:`T`, the corresponding maximum over a
shorter duration :math:`d` can be written as

.. math::
   :label: eq:fbc_companion_distribution

   F_{Q,\max,d}(q) = F_{Q,\max,T}(q)^{d/T}

provided both maxima arise from the same FBC process assumptions.  The
recurrence count belongs to the underlying stochastic process, not to the
load-combination factor itself.

Turkstra's rule is a practical approximation for combining variable actions:
each variable action is taken as the leading action in turn, usually as a
maximum over the reference period, while the other variable actions are taken
as companion values over a representative interval.  In an FBC setting a
companion action can be the point-in-time value or the maximum over the
leading action's basic interval.  The modelling distinction is important:
the FBC process defines the distribution of each action over time; Turkstra's
rule defines which distributions are placed together in each reliability
case.  This is the convention followed in Sørensen's notes and common load
combination examples [Sorensen2004]_ [Faber2009]_.

In Pystra, :class:`~pystra.fbc.FbcProcess` exposes the process distributions:
``point_in_time()`` returns the basic-interval parent distribution, and
``maximum(duration=...)`` returns a maximum distribution for the requested
duration.  :meth:`~pystra.loadcomb.LoadCombination.turkstra` then uses those
process objects to create explicit named leading-action cases.  The result is
still an ordinary :class:`~pystra.loadcomb.LoadCombination`; the generated
cases simply make the FBC and Turkstra assumptions visible in the model.

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


Design Decision Optimization and Societal Risk Acceptance
=========================================================

Design decision optimization with societal risk acceptance is normally applied
after a reliability analysis has estimated :math:`p_f` or :math:`\beta`.  It
does not require a different FORM, SORM, or simulation model.  Instead, an
economic objective is evaluated subject to a societal acceptability criterion.
This follows the risk-based decision framing used in the JCSS risk assessment
guidance [JCSS2008RiskAssessment]_ [KroonMaes2008RiskFramework]_ and the
risk-informed decision principles codified in ISO 2394 [ISO2394]_.
Pystra's initial DDO criterion uses the life quality index (LQI) to define a
minimum acceptable life-safety level with societal willingness to pay (SWTP) as
the life-safety valuation, following the LQI method and the JCSS
risk-assessment background documents and examples [Nathwani1997LQI]_
[Nathwani2009LifeQuality]_ [Rackwitz2002LQI]_ [Rackwitz2008LQI]_
[Streicher2008LQI]_ [Schubert2009LQI]_ [VanCoile2019ALARP]_.

For life-safety problems, the LQI literature expresses the societal willingness
to pay (SWTP) to save one statistical life as a function of the gross domestic
product available for risk reduction, the LQI work--leisure parameter, and a
demographic life-time constant; mortality enters through the demographic
constant rather than through that parameter.  In the notation used by Rackwitz,
this is of the form

.. math::
   :label: eq:lqi_swtp

   \mathrm{SWTP}_x = \frac{g}{q} C_x

where :math:`g` is the income or GDP measure available for risk reduction,
:math:`q` is the dimensionless LQI work--leisure (income-elasticity) parameter
(typically about 0.1--0.2, e.g. 0.175 in Schubert and Faber, 2009; it is *not*
an annual mortality rate), and :math:`C_x` depends on the mortality reduction
scheme, discounting, and the predictive cohort life table
[Rackwitz2004Discounting]_.  In :meth:`~pystra.ddo.Swtp.from_lqi` this parameter
is named ``work_leisure_parameter``.  This SWTP interpretation is developed in the LQI literature
[PandeyNathwani2004LQI]_ [PandeyNathwaniLind2006LQI]_ and used by
Rackwitz for structural reliability optimization and acceptability
[Rackwitz2002LQI]_.  The ``ra.Swtp.from_lqi`` helper
(:meth:`~pystra.ddo.Swtp.from_lqi`) implements the relationship for
user-supplied demographic values.

Pystra also includes a small source-backed country table from Rackwitz's JCSS
background document [Rackwitz2008LQI]_.  The anchor values are the
:math:`G_{\Delta \bar{l}}` column in Table 7, in millions of 1999 PPPUS$:

.. list-table:: Built-in SWTP country values
   :header-rows: 1

   * - Code
     - Country
     - SWTP [10\ :sup:`6` PPPUS$]
   * - CA
     - Canada
     - 1.8
   * - US
     - USA
     - 2.1
   * - AT
     - Austria
     - 1.9
   * - BE
     - Belgium
     - 2.4
   * - CZ
     - Czech Republic
     - 0.54
   * - DK
     - Denmark
     - 1.7
   * - FI
     - Finland
     - 1.3
   * - FR
     - France
     - 1.9
   * - DE
     - Germany
     - 1.9
   * - IT
     - Italy
     - 1.8
   * - NL
     - Netherlands
     - 2.8
   * - NO
     - Norway
     - 1.8
   * - ES
     - Spain
     - 1.3
   * - SE
     - Sweden
     - 1.5
   * - CH
     - Switzerland
     - 1.8
   * - GB
     - United Kingdom
     - 1.7
   * - JP
     - Japan
     - 1.3
   * - NZ
     - New Zealand
     - 1.3

The anchor table is intentionally not overwritten with newer values.  For
current studies, :func:`~pystra.ddo.index_swtp_record` and
``swtp_table(indexed=True)`` return a separately traceable indexed table.
High-level country-based LQI construction requires users to choose
``indexed=True`` or ``indexed=False`` explicitly.  The built-in indexed view
uses the World Bank WDI GDP per capita PPP indicator ``NY.GDP.PCAP.PP.CD`` to
scale each Rackwitz anchor value from 1999 to 2024 [WorldBankWDI]_.  This is a
practical update of the LQI income term :math:`g`; it is not a substitute for a
full recalculation of mortality tables, discounting, and age averaging.

Target reliabilities may be taken from published tables or calculated from an
explicit decision model.  Rackwitz [Rackwitz2000CodeMaking]_ formulates the
code-making problem as an economic optimization in which the annual failure
rate is the natural reliability measure.  Steenbergen, Rózsás, and
Vrouwenvelder [Steenbergen2018Target]_ revisit the same framework and
emphasize annual failure rates as a way to compare targets across design and
remaining working lives.  In normalized form, the Rackwitz/Steenbergen model
implemented by :class:`~pystra.ddo.RackwitzTargetModel` maximizes

.. math::
   :label: eq:rackwitz_target_objective

   Z(p) = B - C(p)
          - U\frac{\lambda}{\gamma} P_{f,\mathrm{SLS}}(p)
          - (C(p)+A)\frac{\omega}{\gamma}
          - (C(p)+H)\frac{\lambda}{\gamma} P_{f,\mathrm{ULS}}(p)

where :math:`p = E[R]/E[S]`, :math:`C(p)=C_0+C_1p`, :math:`\gamma` is the
discount or interest rate, :math:`\omega` is the obsolescence rate, and
:math:`\lambda` is the load occurrence rate.  Pystra uses a closed-form
lognormal resistance-demand model for :math:`P_f(p)` in this calibration.

Every cost is normalized by the base construction cost :math:`C_0`
(``base_cost``), so the model's cost inputs are *ratios* to :math:`C_0`, mapped
to :class:`~pystra.ddo.RackwitzTargetModel` parameters as follows.

.. list-table:: Normalized cost inputs (fractions of :math:`C_0`)
   :header-rows: 1

   * - Symbol
     - Parameter
     - Meaning
   * - :math:`C_1/C_0`
     - ``safety_cost_ratio``
     - marginal safety cost per unit of :math:`p`
   * - :math:`H/C_0`
     - ``failure_cost_ratio``
     - failure (ULS) consequence cost
   * - :math:`U/C_0`
     - ``serviceability_cost_ratio``
     - serviceability (SLS) cost
   * - :math:`A/C_0`
     - ``demolition_cost_ratio``
     - demolition / obsolescence cost
   * - :math:`b/C_0`
     - ``benefit_rate``
     - constant annual benefit (independent of :math:`p`)

The rates :math:`\gamma`, :math:`\omega`, and :math:`\lambda` are
``interest_rate``, ``obsolescence_rate``, and ``load_occurrence_rate``.  Because
the objective is normalized by :math:`C_0`, the resulting target table depends
only on these relative cost and consequence ratios, not on a particular
jurisdiction; the calibrated classes can be compared with the rounded target
reliabilities tabulated in the JCSS Probabilistic Model Code [JCSSPMC2001]_ and
ISO 2394 [ISO2394]_.  A table for other classes is recalculated by passing
``safety_costs`` (the :math:`C_1/C_0` values) and ``failure_costs`` (the
:math:`H/C_0` values) to
:meth:`~pystra.ddo.RackwitzTargetModel.table`.

Fischer, Barnardo, and Faber [Fischer2012LQI]_ provide a convenient LQI route
for turning an SWTP value and expected fatalities given failure into minimum
target reliabilities; the marginal life-saving cost underlying this route is
examined in detail by Fischer et al. [Fischer2013MarginalCost]_.  For medium
variability, define the safety cost ratio

.. math::
   :label: eq:lqi_k1

   K_1 = \frac{C_1(\gamma_S + \omega)}
              {\mathrm{SWTP}\,N_F}

where :math:`C_1(\gamma_S + \omega)` is the marginal safety cost term and
:math:`N_F` is the expected number of fatalities given failure.  The
corresponding target classes are approximated as:

.. list-table:: LQI target reliability classes for medium variability
   :header-rows: 1

   * - Safety cost ratio :math:`K_1`
     - Cost class
     - :math:`\beta`
     - :math:`p_f`
   * - :math:`10^{-3}` to :math:`10^{-2}`
     - Large
     - 3.1
     - :math:`10^{-3}`
   * - :math:`10^{-4}` to :math:`10^{-3}`
     - Medium
     - 3.7
     - :math:`10^{-4}`
   * - :math:`10^{-5}` to :math:`10^{-4}`
     - Small
     - 4.2
     - :math:`10^{-5}`

For normal studies, ``ra.Lqi`` (:class:`~pystra.ddo.Lqi`) builds this target
directly from a country SWTP value or a user-supplied SWTP value, expected
fatalities given failure or an explicit consequence model, and marginal safety
cost.  ``ra.Lqi.lookup_target`` returns the rounded source-table target for a
given :math:`K_1`, while the lower-level :func:`~pystra.ddo.lqi_k1` and
:func:`~pystra.ddo.lqi_target_reliability` helpers remain available in
:mod:`pystra.ddo`.  When the underlying resistance-demand model should be
calculated instead of looked up, ``ra.Lqi.derive_target`` solves the marginal
target problem

.. math::
   :label: eq:lqi_marginal_target

   \min_p\; K_1p + P_f(p),
   \qquad\mathrm{or}\qquad
   K_1 = -\frac{dP_f(p)}{dp}.

``ra.Ddo`` (:class:`~pystra.ddo.Ddo`) then evaluates the selected objective and
criterion without changing the underlying stochastic model.  The best feasible
alternative is obtained with ``Ddo.optimize()``; the unconstrained economic
optimum is available separately as ``Ddo.economic_optimum()``.

For direct JCSS-style optimization, the canonical life-safety risk-cost term
is

.. math::
   :label: eq:lqi_jcss_risk_cost

   S(p) = C(p) + \mathrm{SWTP}\,N_F\,h(p)

where :math:`C(p)` is the safety or construction cost and :math:`h(p)` is a
failure rate or annual failure probability.  The marginal acceptance condition
is

.. math::
   :label: eq:lqi_jcss_acceptance

   \frac{dC(p)}{dp} \ge
   -\mathrm{SWTP}\,N_F\,\frac{dh(p)}{dp}.

``ra.Lqi`` exposes these operations as methods such as
``risk_cost``, ``acceptability_margin_at``, and ``acceptability_boundary``.
The underlying
:func:`~pystra.ddo.jcss_lqi_risk_cost` and
:func:`~pystra.ddo.jcss_lqi_acceptability` functions remain available for
direct reproduction of the JCSS equations.

The current implementation separates an objective from an acceptability
criterion and reserves solver logic for future work.  This keeps LQI in its
proper role as a minimum safety criterion rather than the optimizer itself.
Life-cycle cost and utility models based on stochastic renewal processes are a
natural source for future objective implementations [PandeyWangCheng2015Renewal]_.


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

``pystra.active_learning.ActiveLearning`` combines a surrogate with Monte
Carlo classification and sequential true limit-state evaluations. Kriging
follows the AK-MCS approach [Echard2011]_. The separation of surrogate,
reliability estimator, learning function and stopping criterion follows the
framework discussed by [Moustapha2022]_. The complementary review
[TeixeiraNogalOConnor2021]_ surveys the main adaptive metamodel families.
See :doc:`active_learning` for coverage and proposed extensions, and the
:doc:`notebooks/ex_active_learning` tutorial for independent benchmark references.

An initial Latin hypercube design and a fixed normal Monte Carlo candidate
pool are constructed in **independent standard normal coordinates**. The
configured Nataf or Rosenblatt transformation maps only true evaluations to
physical space. Thus Hermite orthogonality is with respect to independent
normals, including when physical marginals are nonnormal or dependent.
A Student-t spherical Nataf space is unsupported; select Rosenblatt instead.

The surrogate is refitted after each enrichment. Previously evaluated
candidates cannot be selected again. The default initial size is
``max(12, 2*n_variables)``; sparse PCE uses ``max(30, 5*n_variables)``.
Dense OLS uses twice its largest total-degree basis size by default.
The final probability estimate uses an independent Monte Carlo population
that never participates in fitting or point selection.

Surrogates and uncertainty
--------------------------

Kriging uses scikit-learn's Matérn 5/2 Gaussian process with response
normalization and a small numerical nugget. Install the optional ``al`` extra.
Optimizer convergence warnings remain visible; they concern hyperparameter
fitting, separately from the reliability stopping status.

``PceSurrogate`` uses normalized probabilists' Hermite polynomials, selecting
sparse terms by hybrid least-angle regression [BlatmanSudret2011]_. The default
candidate degrees are 1 through 5. ``degree`` and ``q_norm`` can each specify
an increasing sequence: every candidate is fitted and the best corrected
leave-one-out error retained. Hyperbolic truncation and ``max_interaction``
limit the candidate dictionary; ``max_terms`` guards against excessive size.

The implementation adapts the local UQLab 2.2.0 routines, with the copyright
and BSD terms retained in ``THIRD_PARTY_NOTICES``. Direct numerical regression
fixtures compare the original UQLab routines under Octave with PySTRA.
SVD solves replace normal-equation inverses; bootstrap indices are sampled
uniformly. ``method="ols"`` retains dense least-squares fitting.

UQLab's centered, normalized path scoring selects a sparse support. A final
OLS fit on the original Hermite columns supplies the mean and corrected LOO
score used to compare degrees/truncations. ``fit_result`` exposes the selected
degree, q-norm, indices, coefficients and candidate error diagnostics.
Optional early stopping can miss an isolated higher-order term: set
``degree_early_stop=False`` and ``q_norm_early_stop=False`` for exhaustive search.

Pairs-bootstrap refits of the **selected sparse support** supply local spread,
following the fast-bootstrap approach of [MarelliSudret2018]_. Selection is
repeated at each enrichment, but held fixed within each bootstrap ensemble.
The reliability loop still enriches one point at a time; batch enrichment and
full bootstrap model reselection are separate extensions. Rank-deficient
bootstrap draws use minimum-norm least squares, as in UQLab, and their count
is exposed in ``fit_result.n_rank_deficient_bootstrap``. This makes a weak
resampled design visible without conditioning the bootstrap on full rank.

Bootstrap spread is not a Gaussian posterior, a calibrated confidence band,
or a bound on polynomial truncation bias. A common bias across all bootstrap
fits can produce confidently wrong classifications, particularly for nonsmooth
series-system surfaces. Use independent true evaluations, polynomial-degree
checks and benchmark comparisons before trusting a PCE reliability estimate.

Learning functions
------------------

The U-function [Echard2011]_ selects the smallest value of
:math:`U=|\mu|/\sigma`, stopping when its minimum reaches the configurable
threshold (default 2). At zero spread, U is infinite away from the boundary
and zero on it.

The expected feasibility function [Bichon2008]_ selects the largest

.. math::

   \mathrm{EFF} = E[\max(0,\varepsilon-|G|)],
   \qquad G\sim N(\mu,\sigma^2),\quad \varepsilon=2\sigma.

This expectation is symmetric in the mean and nonnegative. Its default
stopping tolerance is :math:`10^{-3}` in **limit-state units**, so rescaling
the limit state requires rescaling this tolerance. Zero spread gives zero
EFF. With PCE, the Gaussian assumption is a heuristic applied to bootstrap
spread; it does not convert that spread into a posterior distribution.

Stopping and interpretation
---------------------------

Learning stops when the configured score threshold is met and the candidate
pool contains both predicted failure and survival. An evaluation budget or
exhausted pool returns explicit nonconvergence. The independent final sample
must also meet ``target_cov`` (default 0.1); otherwise the status is
``sampling_precision``. Zero or all failures never pass this precision check.

The immutable result contains the estimate, normal-equivalent beta,
convergence status, true evaluation count, history, conditional sampling CoV
and an exact 95% binomial interval. These sampling diagnostics exclude
surrogate error. A successful stopping status concerns the sampled points;
it cannot guarantee discovery of disconnected failure regions or eliminate
surrogate bias. Nonconvergence emits a warning and preserves an explicitly
unfinished estimate for diagnosis.


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
