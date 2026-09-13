Adding a distribution
=====================

Marginals inherit from :class:`~pystra.distributions.distribution.Distribution`.
A subclass normally builds a frozen ``scipy.stats`` distribution and passes it
as ``dist_obj`` to the base constructor. The base class supplies density,
CDF, quantile and marginal normal-space transformations. A custom law can
instead implement these operations directly, including both tails: ``pdf``,
``logpdf``, ``cdf``, ``sf``, ``logcdf``, ``logsf``, ``ppf`` and ``isf``.
The marginal ``jacobian(u, x)`` supplies the diagonal derivative ``du/dx``;
the joint transformations expose the directed methods described in
:doc:`/api/probability`.

Continuous transformations require continuous marginals. A reconstruction
contract for ``ZeroInflated`` does not make FORM or smooth copula transforms
valid at its atom. Keep the failure convention ``g <= 0`` when evaluating
mixed distributions through a supported simulation route.

Reconstruction and parameter replacement
----------------------------------------

Every marginal exposes a read-only
:attr:`~pystra.distributions.distribution.Distribution.parameters` mapping.
It contains the complete keyword arguments needed to rebuild the law, including
its name, start point and any bounds, shift, shape or nested distributions::

    import numpy as np
    import pystra as ra

    original = ra.Beta("R", q=2.3, r=4.1, lower=3, upper=10, start_point=5)
    rebuilt = type(original)(**original.parameters)
    copied = original.with_parameters()
    np.testing.assert_allclose(rebuilt.cdf([4, 5, 6]), original.cdf([4, 5, 6]))
    assert copied.start_point == 5

:meth:`~pystra.distributions.distribution.Distribution.with_parameters` returns
an independent instance with selected parameters replaced. It rejects unknown
keywords. Nested marginals and SciPy frozen objects in built-in parameter
snapshots and copies are detached from the original. The mapping itself is
read-only; its values need not be immutable.

Built-ins with native constructors expose native parameters where fitting
moments again would lose precision. For example, Beta exposes ``q``, ``r``,
``lower`` and ``upper``; GEV exposes ``loc``, ``scale`` and ``shape``.
``ScipyDist`` exposes its frozen ``dist_obj``. ``Maximum``, ``MaxParent`` and
``ZeroInflated`` expose their nested marginal and exponent or atom probability.

Moment-parameterized marginals also accept ``mean`` and ``std`` in
``with_parameters``. Supplying either selects moment construction, preserves
the other moment, and keeps fixed bounds or the GEV shape. A native parameter
update keeps the other native parameters. Mixing the two modes raises
``TypeError``. The start point is preserved; explicitly passing
``start_point=None`` selects the new mean::

    changed = original.with_parameters(mean=6, start_point=None)
    assert changed.parameters["lower"] == 3
    assert changed.parameters["upper"] == 10
    assert changed.start_point == changed.mean

A custom marginal whose constructor differs from ``(name, mean, std,
start_point=...)`` must override the public ``parameters`` property.
The mapping must reconstruct the *current* law, and mutable values must be
independent snapshots. For a moment-based subclass with an additional bound::

    from types import MappingProxyType

    @property
    def parameters(self):
        return MappingProxyType({
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "lower": self.lower,
            "start_point": self.start_point,
        })

The inherited ``with_parameters`` passes this mapping, with replacements,
to the constructor. A custom class with alternative parameterizations should
also override ``with_parameters`` to define explicitly which coordinates stay
fixed. It must preserve independence, reject unknown or mixed parameter sets,
and reproduce the law when called without replacements. There is no
``_ctor_kwargs`` or ``_make_copy`` extension hook in 2.0.

Sensitivity coordinates
-----------------------

:attr:`~pystra.distributions.distribution.Distribution.sensitivity_params`
is separate metadata: it names the coordinates to perturb for sensitivity
analysis. The default is ``{"mean": self.mean, "std": self.std}``.
GEV adds ``shape``. Beta bounds participate in reconstruction but remain fixed
during moment sensitivity analysis.

Sensitivity analysis passes the complete sensitivity mapping to
``with_parameters`` and changes one coordinate at a time. Thus a GEV shape
sensitivity holds its mean and standard deviation fixed, even though the
reconstruction mapping uses native parameters::

    load = ra.GEV("S", 10, 2, shape=0.1)
    heavier_tail = load.with_parameters(**{**load.sensitivity_params, "shape": 0.2})
    np.testing.assert_allclose(
        [heavier_tail.mean, heavier_tail.std], [load.mean, load.std]
    )

``ScipyDist`` and the composite marginals have empty sensitivity metadata:
their constructors cannot generically replace a physical mean and standard
deviation. Their copy contract is supported, but ``cdf_gradient`` raises a
clear error until a subclass supplies sensitivity coordinates and derivatives.

The base :meth:`~pystra.distributions.distribution.Distribution.cdf_gradient`
uses central differences. It checks that reconstruction reproduces the law
before perturbing it. Override it with analytic derivatives when available;
the returned mapping must have the same keys as ``sensitivity_params``::

    def cdf_gradient(self, x):
        z = (x - self.mean) / self.std
        density = self.std_normal.pdf(z)
        return {
            "mean": -density / self.std,
            "std": -(x - self.mean) * density / self.std**2,
        }

Verification
------------

Test reconstruction from ``parameters`` and through ``with_parameters``
against moments, support, CDFs, tails and quantiles. Include native constructor
inputs, non-default bounds, explicit start points, and independence of mutable
nested objects. Check that changing a parameter preserves its documented fixed
coordinates and leaves the original unchanged.

For continuous marginals, run a small FORM reference case and compare numerical
and closed-form sensitivities. Tolerances should reflect the finite-difference
step and reference problem; shape derivatives may be less stable than moment
derivatives. See ``tests/test_distribution_copy.py`` and
``tests/test_sensitivity.py`` for examples.
