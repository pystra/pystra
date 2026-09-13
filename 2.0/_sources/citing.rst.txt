Citing PySTRA and its methods
=============================

For a reproducible study, record the PySTRA release, the commit for a development
checkout, Python and optional dependency versions, and the model assumptions.
Use ``pystra.__version__`` to inspect the installed release and ``git rev-parse
HEAD`` in a checkout to identify the commit.

A software reference can name **The PySTRA Developers**, **PySTRA: Python
Structural Reliability Analysis**, the version or commit used, and the
`project repository <https://github.com/pystra/pystra>`_. Include the date you
accessed a development version. The repository's ``LICENSE`` gives the
GPL-3.0-or-later terms and retained notices.

Cite the primary papers for the algorithms and benchmarks used in the study
as well as the software. :doc:`references` collects these sources; the
:doc:`benchmarks` catalogue connects example problems to their papers and
independent checks. PySTRA builds on PyRe, FERUM 4.1 and FERUM; their provenance
and references are retained in the repository README.
