.. _chap_mcmc:

**********************
Theoretical Background
**********************

Reliability
===========

Structural reliability analysis is concerned with the rational treatment of
uncertainties [Melchers1999]_. These uncertainties could be broadly grouped
into three. Thoft- Christensen and Baker (1982) classified them into physical
uncertainties, statistical uncertainties and model uncertainties.

The order of the listed uncertainties corresponds approximately to the
decreasing level of current knowledge and available theoretical tools for
their description and consideration in design. Most of the uncertainties can
never be eliminated absolutely and must be taken into account by engineers
when designing any construction work [Melchers1999]_.


First-Order Reliability Method (FORM)
-------------------------------------

First-Order Reliability Method (FORM) aims at using a first-order
approximation of the limit-state function in the standard space at the
so-called Most Probable Point (MPP) of failure :math:`P^*` (ordesign point),
which is the limit-state surface closest point to the origin. Finding the
coordinates :math:`{\bf u}^*` of the MPP consists in solving the following
constrained optimization problem:

.. math::
   {\bf u} = \arg\min\left\{\|{\bf u}\| \left|g({\bf x}({\bf u}),{\bf \theta}_g) = G({\bf u},{\bf \theta}_g) = 0 \right. \right\} 


Once the MPP :math:`P^*` is obtained, the Hasofer and Lind reliability index
:math:`\beta` is computed as :math:`\beta = {\bf \alpha}^T {\bf u}^*`
where :math:`{\bf \alpha} = -\nabla_u G({\bf u}^*) / \| \nabla_u G({\bf u}^*)\|` 
is the negative normalized gradient vector at the MPP P*. It represents the
distance from the origin to the MPP in the standard space. The first-order
approximation of the failure probability is then given by :math:`p_{f1} =
\Phi(-\beta)`, where :math:`\Phi(\cdot)` is the standard normal cdf. The same
technique is applied to step size evaluation with Armijo rule, where all
corresponding g-calls are sent simultaneously. [Bourinet2010]_

.. _FORM:

.. figure:: _images/FORM.*
   :alt: First-Order Reliability Method.
   :align: center
   :scale: 50



Armijo Rule
-----------

Denote a univariate function :math:`\phi` restricted to the direction
:math:`\mathbf{p}_k` as :math:`\phi(\alpha)=f(\mathbf{x}_k+\alpha\mathbf{p}_k)`.
A step length :math:`\alpha_k` is said to satisfy the Wolfe conditions if the
following two inequalities hold:

.. math::
   i) f(\mathbf{x}_k+\alpha_k\mathbf{p}_k)\leq
      f(\mathbf{x}_k)+c_1\alpha_k\mathbf{p}_k^{\mathrm T}\nabla
      f(\mathbf{x}_k)

.. math::
   ii) \mathbf{p}_k^{\mathrm T}\nabla f(\mathbf{x}_k+\alpha_k\mathbf{p}_k)
       \geq c_2\mathbf{p}_k^{\mathrm T}\nabla f(\mathbf{x}_k)


with :math:`0<c_1<c_2<1`. (In examining condition (ii), recall that to ensure
that :math:`\mathbf{p}_k` is a descent direction, we have
:math:`\mathbf{p}_k^{\mathrm T}\nabla f(\mathbf{x}_k) < 0` .)

:math:`c_1` is usually chosen to be quite small while :math:`c_2` is much
larger; Nocedal gives example values of :math:`c_1=10^{-4}` and
:math:`c_2=0.9` for Newton or quasi-Newton methods and :math:`c_2=0.1` for the
nonlinear conjugate gradient method. Inequality i) is known as the Armijo rule
and ii) as the curvature condition; i) ensures that the step length
:math:`\alpha_k` decreases :math:`f` 'sufficiently', and ii) ensures that the
slope has been reduced sufficiently.


Cholesky decomposition
======================

The Cholesky decomposition of a Hermitian positive-definite matrix A is a
decomposition of the form


.. math::
   \mathbf{A = L L}^{*}


where L is a lower triangular matrix with positive diagonal entries, and L*
denotes the conjugate transpose of L. Every Hermitian positive-definite matrix
(and thus also every real-valued symmetric positive-definite matrix) has a
unique Cholesky decomposition.

If the matrix A is Hermitian and positive semi-definite, then it still has a
decomposition of the form A = LL* if the diagonal entries of L are allowed to
be zero.

When A has real entries, L has real entries as well.

The Cholesky decomposition is unique when A is positive definite; there is
only one lower triangular matrix L with strictly positive diagonal entries
such that A = LL*. However, the decomposition need not be unique when A is
positive semidefinite.

The converse holds trivially: if A can be written as LL* for some invertible
L, lower triangular or otherwise, then A is Hermitian and positive definite.


Gauss base points and weight factors
====================================

using the algorithm given by Davis and Rabinowitz in 'Methods of Numerical
Integration', page 365, Academic Press, 1975.
