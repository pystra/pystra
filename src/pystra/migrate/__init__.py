"""Conservative conversion of PySTRA 1.x source code to the 2.0 API.

Run ``python -m pystra.migrate PATH...`` for unified diffs; add ``--write``
to apply them. Imports, module attributes and resolved constructor keywords
are changed. Diagnostics identify operations requiring manual review.
"""

from ._converter import Conversion, Diagnostic, convert_notebook, convert_source

__all__ = ["Conversion", "Diagnostic", "convert_source", "convert_notebook"]
