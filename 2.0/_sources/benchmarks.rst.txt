Benchmark catalogue
===================

These examples connect a physical or analytical problem to its published source
and an independent check. Open the linked notebook for parameters, citations,
assumptions and executable comparisons. Source notation and reported rounding
are retained where needed. The table describes validation scope, not a ranking
of algorithms.

.. list-table:: Reusable reference problems
   :header-rows: 1
   :widths: 25 35 40

   * - Example
     - Source and assumptions
     - Reference and interpretation
   * - :doc:`notebooks/ex_first_analysis`
     - Independent normal resistance minus load; a teaching problem.
     - Exact normal tail. FORM is exact for the planar normal-space boundary.
   * - :doc:`notebooks/ex_simulation`
     - Hypersphere and parabolic limit states; see [Schueller2007]_ and [Breitung1984]_.
     - Chi-square tail and independent integration; compare sampling methods and local approximation error.
   * - :doc:`notebooks/ex_rosenblatt_system_order`
     - Meinen–Steenbergen's identical-event system with Gaussian or Frank dependence.
     - Direct integration and original-event Monte Carlo. The notebook explains the apparently interchanged dependent-case probabilities in the paper.
   * - :doc:`notebooks/ex_system_reliability`
     - Daniels bundle, component topology, cut sets and Ditlevsen bounds.
     - Published worked values, exact event enumeration and original-system Monte Carlo check different parts of the implementation.
   * - :doc:`notebooks/ex_active_learning`
     - Lognormal beam and four-branch system; active-learning literature benchmarks.
     - Analytical beam probability and independent quadrature for the system; assess classification as well as probability.
   * - :doc:`notebooks/ex_literature_hat`
     - Explicit UQLab hat definition; review comparison in :doc:`literature-benchmarks`.
     - Independent integral and exact cubic representation check. Exact polynomial recovery does not generalize to arbitrary boundaries.
   * - :doc:`notebooks/ex_literature_truss`
     - Published 23-bar truss with ten independent inputs [MarelliSudret2018]_.
     - Independently checked mechanics and conditional integration; distinguish local, sampling and surrogate errors.
   * - :doc:`notebooks/ex_active_extensions`
     - Four-branch system, lognormal beam and two normal tails.
     - PC-Kriging, bootstrap voting and importance sampling checked against independent references.
   * - :doc:`notebooks/ex_active_subset`
     - Linear tail and multiple failure regions.
     - Exact normal-tail probabilities and replicated conditional sampling diagnostics.
   * - :doc:`notebooks/ex_design_decision_optimization`
     - Schubert–Faber JCSS steel-bar decision.
     - Compare published decision points with economic, target-table and marginal LQI criteria.
   * - :doc:`notebooks/ex_target_reliability`
     - LQI and Rackwitz/Steenbergen normalized target models.
     - Compare representative trends with published rounded classes; this does not reproduce every class exactly.

The active-learning overview, extensions and subset-simulation notebooks
require ``pystra[al]`` for Kriging. The hat and truss PCE examples use the core
installation; their runnable bundles include ``literature_benchmarks.py``. Execution cost depends
on training, independent validation and sampling budgets, as well as the
machine and numerical libraries; each notebook states its actual settings.

For detailed source definitions and comparison discrepancies, see
:doc:`literature-benchmarks`.

**Continue:** :doc:`tutorial` · :doc:`guides/methods` · :doc:`references` ·
:doc:`citing`
