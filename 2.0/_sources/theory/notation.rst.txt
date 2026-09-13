Notation and probability spaces
===============================

.. list-table:: Coordinates and reliability quantities
   :header-rows: 1

   * - Symbol
     - Meaning in these docs
   * - :math:`\mathbf{x}`, :math:`\mathbf{X}`
     - Physical input values and random variables, with their engineering units.
   * - :math:`\mathbf{u}`, :math:`\mathbf{U}`
     - Independent standard-normal coordinates in the usual transformation.
   * - :math:`\mathbf{z}`, :math:`\mathbf{Z}`
     - The normal-space notation retained in older derivations; equivalent to u-space there.
   * - :math:`g(\mathbf{x})`, :math:`p_f`
     - Limit-state function and probability of the specified failure event.
   * - :math:`\beta=-\Phi^{-1}(p_f)`
     - Normal-equivalent reliability index.

Subscripts and symbols in reproduced papers retain the source's definitions.
In the continuous examples, failure is described by :math:`g\leq0`; strict and
nonstrict inequalities have the same probability when the boundary has zero
probability. Check the event convention when introducing point masses.

A normal-equivalent index can summarize any probability. It equals the signed
FORM design-point distance when the approximation uses independent normal
coordinates. For explicit generalized Nataf with a Student-t copula, the
standard coordinates are spherical Student-t: their components share a mixing
variable and are dependent (with zero cross-correlation when second moments
exist). The half-space tail is Student-t, so geometric distance and
normal-equivalent beta generally differ. Ordinary Rosenblatt with that same
copula instead produces independent normal coordinates.

The probability integral transform also uses uniforms on the unit hypercube
as intermediate coordinates. A change of coordinates preserves the physical
event probability when the complete joint law is held fixed; a local FORM
approximation need not be invariant under a nonlinear change of coordinates.
See :doc:`/notebooks/ex_rosenblatt_system_order` for the practical consequence.

**Continue:** :doc:`transformations` · :doc:`/copulas` · :doc:`/guides/results`
