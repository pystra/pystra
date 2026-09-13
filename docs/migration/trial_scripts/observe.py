"""Run a trial script and record completed analyses without changing inputs.

Usage: PYTHONPATH=<selected checkout>/src python observe.py SCRIPT OUTPUT.json
"""

from functools import wraps
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pystra as ra
import scipy

records = []
depth = 0
legacy = ra.__version__.startswith("1.")


def observe(cls, label):
    original = cls.run

    @wraps(original)
    def run(self, *args, **kwargs):
        global depth
        depth += 1
        try:
            result = original(self, *args, **kwargs)
            if depth == 1:
                record = {"method": label}
                if label == "SensitivityAnalysis":
                    if legacy:
                        record["marginal"] = result.get("marginal", result)
                    else:
                        record["marginal"] = result.marginal
                elif label == "SORM":
                    record.update(
                        beta=self.betag_breitung if legacy else result.beta,
                        probability=(
                            self.pf2_breitung if legacy else result.failure_probability
                        ),
                        fit=self.fit_type if legacy else result.fit,
                    )
                elif label != "DistributionAnalysis":
                    record.update(
                        beta=self.getBeta() if legacy else result.beta,
                        probability=(
                            self.getFailure() if legacy else result.failure_probability
                        ),
                    )
                    if label == "FORM":
                        record["differentiation"] = (
                            self.options.getDiffMode()
                            if legacy
                            else self.options.differentiation
                        )
                    else:
                        record["cov"] = (
                            float(self.cov_q_bar[self.k - 1])
                            if legacy and hasattr(self, "cov_q_bar")
                            else (
                                float(self.cov)
                                if legacy
                                else result.coefficient_of_variation
                            )
                        )
                records.append(record)
            return result
        finally:
            depth -= 1

    cls.run = run


for old, current in [
    ("Form", "FORM"),
    ("Sorm", "SORM"),
    ("CrudeMonteCarlo", "CrudeMonteCarlo"),
    ("ImportanceSampling", "ImportanceSampling"),
    ("LineSampling", "LineSampling"),
    ("SubsetSimulation", "SubsetSimulation"),
    ("SensitivityAnalysis", "SensitivityAnalysis"),
    ("DistributionAnalysis", "DistributionAnalysis"),
]:
    observe(getattr(ra, old if legacy else current), current)


def plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    array = np.asarray(value)
    return array.item() if array.size == 1 else array.tolist()


calibration = []
if legacy:
    original_design_beta = ra.Calibration.calc_beta_design_param

    @wraps(original_design_beta)
    def calibration_beta(self, design_z):
        result = original_design_beta(self, design_z)
        calibration.append(
            {
                "designs": np.asarray(self.get_design_param_factor()).ravel().tolist(),
                "beta": np.asarray(result).ravel().tolist(),
            }
        )
        return result

    ra.Calibration.calc_beta_design_param = calibration_beta

np.random.seed(20260913)
namespace = runpy.run_path(sys.argv[1], run_name="__main__")
payload = {
    "pystra": ra.__version__,
    "pystra_file": ra.__file__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "records": plain(records),
    "calibration": plain(
        calibration if legacy else namespace.get("trial_calibration", [])
    ),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n")
