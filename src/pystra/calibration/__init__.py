"""Code calibration through normalized reliability and factor verification.

CodeCalibration evaluates candidate factors on normalized code designs.
The solve/derive/select/verify operations derive candidate partial and
combination factors from representative cases and check their resulting designs.
"""

from .normalized import (
    CodeFactors,
    NominalValues,
    NormalizedReliabilityModel,
    CodeCalibration,
    CodeDesignResult,
    CodeCalibrationResult,
)
from .factors import (
    FactorCalibrationProblem,
    CalibratedDesign,
    TargetDesigns,
    FactorSet,
    GoverningFactor,
    DesignValues,
    DesignVerification,
    analyze_case,
    solve_designs,
    derive_factors,
    select_factors,
    design_with_factors,
    verify_designs,
)
from .plotting import plot_calibration

__all__ = [
    "CodeFactors",
    "NominalValues",
    "NormalizedReliabilityModel",
    "CodeCalibration",
    "CodeDesignResult",
    "CodeCalibrationResult",
    "FactorCalibrationProblem",
    "CalibratedDesign",
    "TargetDesigns",
    "FactorSet",
    "GoverningFactor",
    "DesignValues",
    "DesignVerification",
    "analyze_case",
    "solve_designs",
    "derive_factors",
    "select_factors",
    "design_with_factors",
    "verify_designs",
    "plot_calibration",
]
