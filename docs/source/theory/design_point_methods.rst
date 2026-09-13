FORM, SORM and design-point diagnostics
***************************************

First-Order reliability method (FORM)
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


.. figure:: ../images/f-02-09-a.*
   :alt: FORM a
   :align: center
   :scale: 50

Representation of a physical space with a set :math:`{\bf X}` of any two
random variables. The shaded area denotes the failure domain and
:math:`g({\bf X}) = 0` the failure surface.

.. figure:: ../images/f-02-09-b.*
   :alt: FORM b
   :align: center
   :scale: 50

After
transformation in the normalized space, the random variables :math:`{\bf X}`
are now uncorrelated and standardized normally distributed, also the failure
surface is transformed into :math:`g({\bf Z}) = 0`.

.. figure:: ../images/f-02-09-c.*
   :alt: FORM c
   :align: center
   :scale: 50

FORM corresponds to a linearization of the failure surface :math:`g({\bf Z}) =
0`. Performing this method, the design point :math:`{\bf z}^*` and the
reliability index :math:`\beta` can be computed.


Second-Order reliability method (SORM)
======================================

FORM approximates the failure surface :math:`g({\bf Z}) = 0` by a tangent
hyperplane at the design point.  When the failure surface has significant
curvature at the design point, this linear approximation can over- or
under-estimate :math:`p_f`.  The Second-Order Reliability Method (SORM)
improves on FORM by fitting a quadratic surface (paraboloid) to
:math:`g({\bf Z}) = 0` at the design point, thereby capturing
second-order effects [Baker2010]_.

Quadratic approximation in rotated space
----------------------------------------

Starting from the FORM design point :math:`{\bf z}^*` and the unit
direction vector :math:`\boldsymbol{\alpha} = -{\bf z}^*/\beta`, the
standard normal space is rotated so that :math:`{\bf z}^*` lies at
distance :math:`\beta` along the last axis.  Let :math:`{\bf R}` denote
the orthonormal rotation matrix constructed by Gram--Schmidt
orthonormalization with :math:`\boldsymbol{\alpha}` in the last row, and
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
PySTRA provides two approaches.

Curve-Fitting
-------------

The default method (``fit="curve"``) obtains the curvatures from the
Hessian matrix of the limit state function.  The Hessian :math:`{\bf H}`
of :math:`g` at the design point :math:`{\bf z}^*` is computed by finite
differences of the gradient that is already available from FORM.  This
matrix is then rotated and normalized:

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

An alternative method (``fit="point"``) determines the curvatures by
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
:math:`\kappa_i^-`.  The generalized Breitung formula for asymmetric
curvatures is:

.. math::
    :label: eq:sorm_breitung_pf

    p_{f2} = \Phi(-\beta) \prod_{i=1}^{n-1} \frac{1}{2}
    \left[ \left(1 + \beta\, \kappa_i^+\right)^{-1/2}
         + \left(1 + \beta\, \kappa_i^-\right)^{-1/2} \right]

When the curvatures are symmetric (:math:`\kappa_i^+ = \kappa_i^-`), this
reduces to the standard Breitung formula :eq:`eq:sorm_breitung`.

Hohenbichler--Rackwitz Modification
-----------------------------------

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
Both the standard and modified Breitung results are reported by PySTRA.

Validity and method comparison
------------------------------

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


.. _theory_strong_maximum:
.. _/theory.rst#theory-strong-maximum:

Strong Maximum Test
===================

A converged local FORM design point need not represent every important
failure region. The Strong Maximum Test [DutfoyLebrun2006]_ probes an enlarged
sphere around the origin for failure points outside the candidate's vicinity.
PySTRA implements the independent standard-normal case described by
`OpenTURNS <https://openturns.github.io/openturns/latest/theory/reliability_sensitivity/strong_maximum_test.html>`_.
See :doc:`/notebooks/ex_strong_maximum` for geometric examples and
:ref:`chap_strong_maximum` for the API.

Sphere geometry
---------------

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
PySTRA classifies failure by :math:`g<0`. Far failure points are possible
restart locations for additional design-point searches, not optimized design
points themselves. The magnitude of :math:`g` does not measure a region's
probability importance.

Cap probability and evaluation budget
-------------------------------------

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

PySTRA rounds upward to meet the requested nominal confidence; OpenTURNS'
reference implementation rounds to the nearest integer. Users can specify
confidence or a fixed count, with a hard ``max_points`` budget checked before
sphere evaluation. The total is :math:`N+2` point evaluations including the
origin and boundary checks. The cap can become small in high dimensions,
so inspect the budget before using an expensive structural model.

Interpretation and limitations
------------------------------

Nominal confidence is a sampling statement about hitting a fixed cap under
the test's local-plane and failure-region extent assumptions. It is not a
posterior probability that FORM is correct, a failure-probability estimate,
or a bound on approximation error. A bounded failure island entirely inside
the sphere cannot be detected, regardless of sample count. The tutorial
constructs such an island closer to the origin than the supplied candidate.

Use the diagnostic on individual ``SystemFORM.component_results`` to look
for missed regions within each component. Checking every component does not
validate the system probability. With a non-Gaussian copula, use Rosenblatt
so that the sphere geometry and cap probability apply in independent normal
space. Generalized Student-t Nataf would require different density-radius
geometry and is not supported by this implementation.

**Use this method:** :doc:`/guides/form_sorm` · :doc:`/notebooks/ex_intro` · :doc:`/api/reliability`

For coordinate conventions, see :doc:`notation`.
