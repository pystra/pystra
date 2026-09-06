"""Code-design studies and specialist design-point factor calibration.

GenericCalibration evaluates candidate factors on normalized code designs.
The separate solve/derive/select/verify operations preserve the traditional
factor-calibration methods without mutable orchestration tables.
"""

from .generic import (
    CodeFactors,
    NominalValues,
    GenericModel,
    GenericCalibration,
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
    "GenericModel",
    "GenericCalibration",
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
