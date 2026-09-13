Models and analysis options
===========================

Define random variables, limit states, model constants and algorithm options.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :class:`~pystra.model.StochasticModel`
     - Collect named variables, constants and dependence.
   * - :class:`~pystra.model.LimitState`
     - Wrap the physical response and gradient contract.
   * - :mod:`pystra.options`
     - Frozen FORM, SORM and simulation settings.

**Use it:** :doc:`/guides/models` · :doc:`/notebooks/ex_first_analysis` · :doc:`/theory/fundamentals`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.model
   pystra.reliability.analysis
   pystra.options
   pystra.errors

Limit-state evaluation
----------------------

``LimitState.evaluate(x, model, differentiation="ffd")`` accepts a physical
point ``(n_variables,)`` or rows of points ``(n_samples, n_variables)``.
Values are scalar for a point and ``(n_samples,)`` for a batch; gradients
have the input shape. Columns and gradient entries follow model random-variable
order, excluding constants. Square batches always contain rows of points.
``differentiation="ddm"`` uses the expression's analytic gradient;
``"no"`` returns zero gradients. Failure is ``g <= 0``.

User-script migration
---------------------

``python -m pystra.migrate path/to/script.py path/to/notebook.ipynb`` previews
unified diffs for 1.x code. ``--write`` applies the same changes. Directories
are scanned for Python files and notebooks. The converter preserves local
identifiers, string literals and comments, and changes only resolved PySTRA
imports, module attributes and constructor keywords. Shadowed names remain
unchanged. Instance methods, result getters, options, parameter modes and
calibration workflows require manual review; diagnostics identify their
locations. A second pass must leave the source unchanged.

Notebook imports carry across valid Python cells in document order. Cells
containing IPython syntax are left unchanged and flagged. Only changed code
source fields are serialized; Markdown, outputs and metadata are preserved.
Exit status is zero with no review items, one when manual review is needed,
and two for file or input errors. A successful conversion is not a guarantee
that the resulting analysis is runnable or numerically appropriate.
