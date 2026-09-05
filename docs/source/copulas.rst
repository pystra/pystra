.. _chap_copulas:

Copulas and Isoprobabilistic Transformations
=================================================

For worked examples, see :doc:`notebooks/ex_copulas`; for the derivations,
see :ref:`theory_copulas`.

A joint distribution has two separate ingredients: marginal distributions
and a copula describing their dependence. Choosing a transformation for
reliability analysis does not change that joint distribution. This follows
the distinction developed by Lebrun and Dutfoy [LebrunDutfoy2009a]_
[LebrunDutfoy2009b]_.

Specifying a joint distribution
------------------------------------

.. code-block:: python

   import pystra as ra

   joint = ra.JointDistribution(
       [ra.Lognormal("R", 10.0, 2.0), ra.Normal("S", 4.0, 1.0)],
       ra.StudentTCopula([[1.0, 0.4], [0.4, 1.0]], df=4),
   )
   model = ra.StochasticModel(joint)
   model.addVariable(ra.Constant("C", 1.0))
   samples = joint.rvs(1000, seed=2026)  # rows of observations, columns R, S
   print(joint.pdf([10.0, 4.0]))
   print(joint.cdf([10.0, 4.0]))

Alternatively, add all random variables to an existing model and then call
``model.setCopula(copula)``. Constants may be added afterwards. The copula's
dimension must equal the number of random variables. Named continuous Pystra
marginals, including ``ScipyDist``, are supported; mixed/discrete marginals
are outside this implementation's scope. Known zero-inflated marginals with
positive point mass are rejected.

The initial families are:

* ``IndependentCopula(dimension)``: the product copula.
* ``GaussianCopula(R)``: ``R`` is the latent normal correlation.
* ``StudentTCopula(R, df)``: ``R`` is the latent Student-t shape matrix with
  unit diagonal. Positive degrees of freedom are supported, including values
  for which the latent representative has no covariance. Identity ``R``
  does **not** give independence, because all coordinates share a random scale.
* ``FrankCopula(theta)``: a bivariate non-elliptical example; zero gives
  independence. The initial numerical implementation supports ``abs(theta)
  <= 30``.

Gaussian and Student-t matrices must be symmetric and positive definite.
They can also be constructed using ``from_kendall_tau(tau, ...)``, with
``df`` supplied for Student-t. This uses
:math:`R_{ij}=\sin(\pi\tau_{ij}/2)` and validates the resulting matrix.
These matrices generally differ from the physical marginals' Pearson
correlations. Copula fitting and physical Pearson calibration for Frank or
Student-t copulas are not included.

Existing ``model.setCorrelation(...)`` retains its physical Pearson meaning
and the existing Gaussian Nataf calibration. Setting a copula explicitly
replaces that specification; setting a correlation explicitly switches back
to the legacy specification. ``getCorrelation()`` raises for an explicit
copula to prevent confusion between physical and latent correlations. Use
``getCopula()`` for the dependence specification. ``getJointDistribution()``
also works for a legacy model by calibrating its Gaussian copula.

Choosing the transformation
--------------------------------

.. code-block:: python

   options = ra.AnalysisOptions()
   options.setTransform("rosenblatt")
   options.setRosenblattOrder([1, 0])  # condition on S before R
   form = ra.Form(
       stochastic_model=model,
       analysis_options=options,
       limit_state=ra.LimitState(lambda R, S, C: R - C*S),
   )
   form.run()
   print(form.getBeta(), form.getFailure())

Rosenblatt uses sequential conditional CDFs followed by the normal quantile
function. Its standard coordinates are independent standard normals for
every supported copula. Returned coordinate arrays retain original variable
indices: ``u[order[k]]`` is the k-th conditional innovation. Every component
of a system must use the same order.

With default analysis options, an explicit Gaussian copula uses Nataf;
Student-t and Frank use Rosenblatt. Existing models without an explicit
copula retain their previous Gaussian Nataf behaviour.

``options.setTransform("nataf")`` selects generalized Nataf for an elliptical
copula. ``"cholesky"`` and ``"svd"`` select Nataf factorisations explicitly;
these choices are unavailable for Frank. Conditioning order applies only to
Rosenblatt. Gaussian Nataf with Cholesky factorisation equals Rosenblatt in
the same order. Different Gaussian orders are related by an orthogonal
change of normal coordinates, so the optimum FORM probability is invariant.
For non-Gaussian copulas, changing Rosenblatt order can change the FORM
approximation, although it cannot change the exact failure probability.

Transformations are also available independently of an analysis:

.. code-block:: python

   transform = joint.getTransformation("rosenblatt", order=[1, 0])
   u = transform.x_to_u([10.0, 4.0])
   x = transform.u_to_x(u)
   du_dx = transform.jacobian(u, x)

These methods take one full vector; joint distribution methods take either
one point or rows of points. Copula-level ``rosenblatt`` and
``inverse_rosenblatt`` operate on **uniform** coordinates, whereas the joint
transformation uses normal coordinates. Elliptical transformation Jacobians
are analytic; Frank uses central differences of the inverse transformation,
without extra limit-state evaluations.

Generalized Nataf is not always normal
-------------------------------------------

For an elliptical copula, generalized Nataf first maps each marginal into
the elliptical representative and then removes its shape matrix. The result
is spherical, but is independent normal only for a Gaussian copula.

For ``StudentTCopula(R, df)``, the Nataf coordinates have spherical Student-t
law with identity **shape**, not independent coordinates or unit covariance.
The univariate scale is one; when ``df > 2``, covariance is
:math:`\nu/(\nu-2) I`. FORM therefore uses

.. math::

   P_{f,\mathrm{FORM}} = T_\nu(-\beta),

where :math:`\beta` is the signed geometric distance in this spherical
space. ``form.getBeta()`` returns this distance, while
``form.getEquivalentBeta()`` returns :math:`-\Phi^{-1}(P_f)` for comparison
with conventional normal reliability indices.

SORM, system FORM, simulation methods and the Strong Maximum Test currently
require independent normal coordinates and reject spherical Student-t space.
Use Rosenblatt with these methods. Closed-form parameter sensitivities still
assume the legacy physical-Pearson Nataf specification; for explicit copulas,
use numerical sensitivities, which keep the copula parameters fixed.

Validation and numerical scope
-----------------------------------

The tests compare Gaussian and Student-t joint densities with SciPy's
multivariate distributions, verify transformation Jacobians against density
identities, transform independent reference Student-t samples, and check
an exact affine Student-t FORM half-space probability.

The second paper's exponential benchmark uses rates 1 and 3 and failure
event :math:`8X_1+2X_2-1\leq0`. With Frank parameter 10, the two Rosenblatt
orders reproduce FORM probabilities approximately 0.107 and 0.122; direct
integration gives approximately 0.1038. With Gaussian latent correlation
0.5, both orders and Cholesky Nataf give approximately 0.09758.

Transformations require finite points and resolvable interior probabilities;
they do not silently clip a failed tail calculation into the unit interval.
Gaussian and Student-t joint transformations use survival functions where
available to preserve tail resolution. Copula uniform coordinates still
have floating-point limits: in very low conditional-density regions,
roundoff can be amplified substantially by the inverse map. Marginal
implementations may impose additional tail limits. Joint CDFs for elliptical
copulas use numerical SciPy integration and accept its integration keyword
arguments; they are not exact rare-event probability estimators.
