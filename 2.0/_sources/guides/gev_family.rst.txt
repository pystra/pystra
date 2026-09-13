Choosing an extreme-value distribution
======================================

Use the family name when constructing an extreme-value marginal in PySTRA.
The Type I, II and III names describe the classical families; the maxima or
minima convention determines which tail is bounded. State the reference
period of the extreme, such as an annual maximum load, alongside its units
and parameters.

Match the family and the tail
-----------------------------

.. list-table:: Extreme-value names in PySTRA 2.0
   :header-rows: 1
   :widths: 24 24 27 25

   * - Classical family
     - PySTRA class
     - Tail convention
     - Former 1.x name
   * - Type I, largest
     - :class:`~pystra.distributions.gumbel.Gumbel`
     - Gumbel maxima; unbounded support
     - ``TypeIlargestValue``
   * - Type I, smallest
     - :class:`~pystra.distributions.gumbel.GumbelMin`
     - Gumbel minima; reflected maxima
     - ``TypeIsmallestValue``
   * - Type II, largest
     - :class:`~pystra.distributions.frechet.Frechet`
     - Fréchet maxima; heavy upper tail
     - ``TypeIIlargestValue``
   * - Type III, smallest
     - :class:`~pystra.distributions.weibull.Weibull`
     - Weibull minima; finite lower bound
     - ``TypeIIIsmallestValue``
   * - Generalized extreme value, maxima
     - :class:`~pystra.distributions.gev.GEV`
     - Shape selects Type I, II or III
     - ``GEV``; ``GEVmax`` also remains valid
   * - Generalized extreme value, minima
     - :class:`~pystra.distributions.gev.GEVMin`
     - Reflected GEV maxima
     - ``GEVmin``

``GEVmax`` is a permanent alias for ``GEV``: both names refer to the same
class. The old type-numbered names and ``GEVmin`` have been replaced; see
:doc:`/migrating` for script conversion.

The ordinary Weibull distribution describes minima. A Type III distribution
for maxima has a finite *upper* endpoint and is often called the reversed
Weibull distribution. Construct that case with ``GEV(shape=...)`` using a
negative shape. The Fréchet and Weibull names and their SciPy parameterizations
are documented in `SciPy's invweibull reference
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html>`_
and `weibull_min reference
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html>`_.

Read the GEV shape convention
-----------------------------

Write :math:`\ell` for ``loc``, :math:`s>0` for ``scale`` and
:math:`\xi` for ``shape``. The maxima CDF is

.. math::

   F_{\max}(x) =
   \exp\!\left[-\left(1+\xi\frac{x-\ell}{s}\right)^{-1/\xi}\right],
   \qquad 1+\xi\frac{x-\ell}{s}>0.

At :math:`\xi=0` the limiting expression is
:math:`F_{\max}(x)=\exp[-\exp(-(x-\ell)/s)]`. PySTRA uses the opposite
shape sign to `SciPy's genextreme
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html>`_:
the corresponding SciPy argument is ``c=-shape``.

.. list-table:: Shape and support for GEV maxima
   :header-rows: 1
   :widths: 20 30 50

   * - Shape
     - Family
     - Support
   * - :math:`\xi=0`
     - Type I, Gumbel
     - All real values
   * - :math:`0<\xi<0.5`
     - Type II, Fréchet
     - Lower endpoint :math:`\ell-s/\xi`; unbounded above
   * - :math:`\xi<0`
     - Type III, reversed Weibull
     - Unbounded below; upper endpoint :math:`\ell-s/\xi`

``GEVMin(loc=ell, scale=s, shape=xi)`` is the distribution of
:math:`-Y` when :math:`Y` has
``GEV(loc=-ell, scale=s, shape=xi)``. Reflection changes the CDF to

.. math::

   F_{\min}(x) =
   1-\exp\!\left[-\left(1+\xi\frac{\ell-x}{s}\right)^{-1/\xi}\right].

Its support requires :math:`1+\xi(\ell-x)/s>0`. Thus negative shape gives
a finite lower endpoint :math:`\ell+s/\xi`, and positive shape gives a
heavy lower tail with finite upper endpoint :math:`\ell+s/\xi`.
Zero shape gives ``GumbelMin``.

Both GEV classes require ``shape < 0.5`` for finite variance. Supply either
``mean`` and ``std``, or ``loc`` and ``scale``, together with ``shape``.
Location and scale have the variable's units; shape is dimensionless.
Location generally differs from the mean, and scale generally differs from
the standard deviation.

Check equivalent parameterizations
----------------------------------

The following examples compare quantiles across each named family. For
Fréchet scale :math:`a` and shape :math:`k>2`, the equivalent GEV parameters
are :math:`(\ell,s,\xi)=(a,a/k,1/k)`. For Weibull lower bound :math:`b`,
scale :math:`a` and shape :math:`k>0`, the equivalent GEVMin parameters
are :math:`(\ell,s,\xi)=(b+a,a/k,-1/k)`. These relations follow by
substitution into the CDFs; the family-specific shape is not the GEV shape.

.. testcode:: gev-family

   import numpy as np
   import pystra as ra

   probabilities = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
   pairs = [
       (
           ra.Gumbel("Q", loc=10.0, scale=2.0),
           ra.GEV("Q", loc=10.0, scale=2.0, shape=0.0),
       ),
       (
           ra.GumbelMin("R", loc=10.0, scale=2.0),
           ra.GEVMin("R", loc=10.0, scale=2.0, shape=0.0),
       ),
       (
           ra.Frechet("Q", scale=12.0, shape=4.0),
           ra.GEV("Q", loc=12.0, scale=3.0, shape=0.25),
       ),
       (
           ra.Weibull("R", lower=2.0, scale=8.0, shape=4.0),
           ra.GEVMin("R", loc=10.0, scale=2.0, shape=-0.25),
       ),
   ]
   for family, generalized in pairs:
       np.testing.assert_allclose(
           family.ppf(probabilities),
           generalized.ppf(probabilities),
           rtol=1e-12,
           atol=1e-12,
       )
   assert ra.GEVmax is ra.GEV

Each pair represents the same marginal distribution. Choosing a family from
data still requires checking the support, tail fit and reference period.
For the exact maximum of a specified number of independent, identically
distributed parent variables, see
:class:`~pystra.distributions.maximum.Maximum` and its parent-CDF construction.

**Continue:** :doc:`models` · :doc:`/api/probability` ·
:doc:`/notebooks/ex_scipy_distributions`
