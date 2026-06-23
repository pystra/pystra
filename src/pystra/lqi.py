"""Backward-compatible aliases for :mod:`pystra.ddo`.

New code should import design decision optimization helpers from
``pystra.ddo``.  The LQI criterion and SWTP values remain part of that module.
"""

from .ddo import *  # noqa: F401,F403
from .ddo import plot_decision_summary as plot_lqi_summary
