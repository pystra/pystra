.. _welcome-to-pystra-s-documentation:

PySTRA: structural reliability in Python
========================================

PySTRA implements established and selected modern structural reliability
methods, with practical tools for code calibration and structural assessment.
Define your engineering model in Python, quantify its reliability, and inspect
the assumptions and diagnostics behind the result.

These pages describe **PySTRA** |release|, the **2.0 development API**.
Follow :doc:`install` to install this version. :doc:`whatsnew` summarizes the
release, and existing users can consult :doc:`migrating`; the
`stable documentation <https://pystra.github.io/pystra/>`_ uses the released 1.x
API.

.. container:: workflow-grid

   .. container:: workflow-card

      **Run a reliability analysis**

      Build a two-variable model and check FORM against an analytic answer.

      :doc:`Start your first analysis <notebooks/ex_first_analysis>`

   .. container:: workflow-card

      **Calibrate code factors**

      Compare candidate factors using normalized reliability and verify
      the resulting designs.

      :doc:`Follow the calibration workflow <guides/calibration>`

   .. container:: workflow-card

      **Compare assessment scenarios**

      Evaluate how specified resistance and load assumptions change reliability.

      :doc:`Assess a structure <guides/assessment>`

   .. container:: workflow-card

      **Estimate very small probabilities**

      Tail-accurate transformations and log-space estimators stay finite far
      beyond where floating-point probabilities round off.

      :doc:`Work at high reliability <guides/high_reliability>`

.. _indices-and-tables:

Find the right material
-----------------------

* :doc:`get_started` takes you from installation to a checked first result.
* :doc:`whatsnew` summarizes what 2.0 adds and changes.
* :doc:`user_guide` helps you choose methods, build models and interpret results.
* :doc:`tutorial` contains worked tutorials and published benchmark problems.
* :doc:`api` describes public objects, parameters, results and extension details.
* :doc:`theory` explains the formulations, assumptions and notation.

For method selection, begin with :doc:`guides/methods`. For unexpected results,
see :doc:`guides/troubleshooting`. To reproduce a paper example, browse the
:doc:`benchmarks`.

.. toctree::
   :hidden:
   :maxdepth: 2

   get_started
   whatsnew
   user_guide
   tutorial
   api
   theory
   migrating
   changelog
   references
   developer

Project and community
----------------------

PySTRA is GPL-3.0-or-later. Its numerical methods build on the work cited in
:doc:`references`, including the original framework [Hackl2013]_.
See :doc:`citing` for software citation and method attribution.
Report questions and reproducible problems through the
`issue tracker <https://github.com/pystra/pystra/issues>`_.
