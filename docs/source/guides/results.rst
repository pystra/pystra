Interpreting results and convergence
====================================

Report the event, probability model, method, convergence status and numerical
precision together. A failure probability without its reference period or
modelling assumptions is incomplete.

FORM records
------------

:meth:`pystra.reliability.form.FORM.run` returns an immutable
:class:`~pystra.results.FORMResult`. Its values remain a snapshot if the solver
is subsequently rerun.

.. list-table:: Fields to read first
   :header-rows: 1

   * - Field
     - Meaning
   * - ``converged``, ``message``
     - Whether the numerical convergence criteria were met.
   * - ``failure_probability``, ``beta``
     - The approximation and its normal-equivalent index, :math:`-\Phi^{-1}(p_f)`.
   * - ``design_point``, ``variable_names``
     - Physical coordinates, in the stated variable order and original units.
   * - ``standard_point``, ``standard_space``
     - The same point in the chosen probability coordinates.
   * - ``geometric_beta``
     - Signed distance in those coordinates; equals ``beta`` in normal space.
   * - ``iterations``, ``limit_state_error``, ``direction_error``
     - Diagnostics for the iteration and its termination.

An unconverged record has no probability, reliability index or design point.
The analysis object's ``get_beta()`` returns geometric beta, whereas
``get_equivalent_beta()`` and the record's ``beta`` are normal-equivalent.
This distinction matters for explicit spherical Student-t transformations;
see :doc:`/theory/notation`.

Other analyses
--------------

Result interfaces differ by algorithm. SORM exposes named Breitung and modified
Breitung result attributes after a valid run. Classical simulations expose
``get_failure()`` and ``get_beta()`` on the completed analysis. Use their API
pages for the precise contract; do not assume every ``run()`` returns a
``FORMResult``. Retain the convergence and termination information of system,
calibration and active-learning result records when exporting tables.

Separate three sources of error
-------------------------------

* **Numerical convergence:** did the iteration or fit meet its criteria?
* **Approximation error:** does a tangent plane, quadratic surface or surrogate
  represent the important failure regions?
* **Sampling uncertainty:** how variable is an estimate based on finite samples?

A confidence interval on a frozen surrogate's Monte Carlo estimate addresses
the last item, conditional on that surrogate. It does not measure its bias.
A completed calculation or a small coefficient of variation does not establish
the physical model's adequacy.

**Continue:** :doc:`troubleshooting` · :doc:`/plotting` ·
:doc:`/notebooks/ex_first_analysis` · :doc:`/api/reliability`
