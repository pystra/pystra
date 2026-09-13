Structural reliability fundamentals
===================================

.. _structural-reliability:


Structural reliability analysis quantifies the probability that a structure
fails to satisfy a stated performance requirement, under an explicit model of
uncertain actions, resistance and model error. [Melchers1999]_

.. _limit-states:

Limit states
------------

A limit state specifies the boundary between acceptable and unacceptable
performance. The event must be defined before its probability can be calculated;
different requirements can produce different component or system events.
[Malioka2009]_

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

The classical approach
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

.. figure:: ../images/f-02-07-a.*
   :alt: Overlapping resistance and load densities used in the failure-probability integral.
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

.. figure:: ../images/f-02-08-a.*
   :alt: Resistance, load and safety-margin densities; beta standard deviations separate the margin mean from zero.
   :align: center
   :scale: 50


Hasofer and Lind reliability index
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

.. toctree::
   :hidden:

   notation

**Use this method:** :doc:`/guides/models` · :doc:`/notebooks/ex_first_analysis` · :doc:`/api/models`

For coordinate conventions, see :doc:`notation`.
