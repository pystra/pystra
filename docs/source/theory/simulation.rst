Simulation methods
==================

.. _id1:


Simulation estimates the event probability directly from sampled evaluations,
including when a local design-point approximation is inadequate. The starting
point is the probability integral in Equation :eq:`eq:2_70`. Direct Monte Carlo
is the basic estimator; importance sampling, line sampling and subset simulation
seek to reduce the cost of rare-event estimation. [Nowak2000]_ [Faber2009]_

See :doc:`/guides/simulation` for the practical workflow and
:doc:`design_point_methods` for the local approximations being compared.

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


Crude Monte Carlo simulation
----------------------------

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


Importance sampling
-------------------

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

.. figure:: ../images/f-02-10-a.*
   :alt: MC a
   :align: center
   :scale: 50

Representation of a physical space with a set :math:`{\bf X}` of any two
random variables. The shaded area denotes the failure domain and `g({\bf X}) =
0` the failure surface.

.. figure:: ../images/f-02-10-b.*
   :alt: MC b
   :align: center
   :scale: 50

For the CMC method every dot corresponds to one configuration of the random
variables :math:`{\bf X}`. Dots in shaded areas lead to a failure.

.. figure:: ../images/f-02-10-c.*
   :alt: MC c
   :align: center
   :scale: 50


The IS simulation method uses a distribution centered on the design point
:math:`{\bf x}^*`, is obtained from a FORM (or SORM) analysis. More dots in
the failure domain can be observed.


Line sampling
-------------

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


Subset simulation
-----------------

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

**Use this method:** :doc:`/guides/simulation` · :doc:`/notebooks/ex_simulation` · :doc:`/api/reliability`

For coordinate conventions, see :doc:`notation`.
