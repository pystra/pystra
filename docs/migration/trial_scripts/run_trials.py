"""Reproduce conservative conversion and numerical migration trials.

Run from the checkout with its src on PYTHONPATH. The baseline must be an
unmodified export or detached checkout of v1.6.0. Output includes both versions'
records, execution logs, converter diffs, and reviewed manual patches.
"""

import argparse
from dataclasses import asdict
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from pystra.migrate import convert_notebook, convert_source

ROOT = Path(__file__).resolve().parents[3]
TRIALS = Path(__file__).resolve().parent
SCRIPTS = {
    **{
        name: f"examples/{name}.py"
        for name in (
            "example",
            "ddm_example",
            "gev_example",
            "sensitivity",
            "timing",
            "openseespy_ex",
        )
    },
    **{
        name: f"docs/source/notebooks/{name}.ipynb"
        for name in (
            "ex_intro",
            "ex_ddm",
            "ex_scipy_distributions",
            "ex_sensitivity",
            "ex_simulation",
            "ex_factor_calibration",
        )
    },
}


def code(source, suffix):
    if suffix == ".ipynb":
        return (
            "\n\n".join(
                "".join(cell["source"])
                for cell in json.loads(source)["cells"]
                if cell["cell_type"] == "code"
            )
            + "\n"
        )
    return source


def baseline_adjustments(source, name):
    source = source.replace("number = 100", "number = 1")
    if name == "sensitivity":
        source = source.replace(
            "form.showDetailedOutput()", "form.run()\nform.showDetailedOutput()"
        )
    if name == "ex_simulation":
        source = re.sub(
            r"float\((\w+\.get(?:Failure|Beta)\(\))\)",
            r"float(np.asarray(\1).item())",
            source,
        )
    return source


def flatten(mapping):
    return np.array([value for row in mapping.values() for value in row.values()])


def compare(baseline, current):
    """Compare numerical outputs, keeping distinct stochastic error criteria."""
    checks = []
    if baseline.get("calibration"):
        np.testing.assert_allclose(
            [list(row.values()) for row in current["calibration"]],
            [list(row.values()) for row in baseline["calibration"]],
            rtol=0,
            atol=1e-8,
        )
        return [{"method": "Calibration", "cases": 8, "absolute_tolerance": 1e-8}]
    reference = baseline["records"]
    indices = {}
    for record in current["records"]:
        method = record["method"]
        if method == "DistributionAnalysis":
            continue
        candidates = [row for row in reference if row["method"] == method]
        if method == "FORM":
            candidates = [
                row
                for row in candidates
                if row["differentiation"] == record["differentiation"]
            ]
            previous = min(
                candidates, key=lambda row: abs(row["beta"] - record["beta"])
            )
            np.testing.assert_allclose(
                [record["beta"], record["probability"]],
                [previous["beta"], previous["probability"]],
                rtol=1e-10,
                atol=1e-10,
            )
            checks.append({"method": method, "absolute_tolerance": 1e-10})
        else:
            index = indices.get(method, 0)
            previous = candidates[index]
            indices[method] = index + 1
            if method == "SORM":
                np.testing.assert_allclose(
                    record["beta"], previous["beta"], rtol=0, atol=2e-4
                )
                np.testing.assert_allclose(
                    record["probability"], previous["probability"], rtol=1e-3, atol=0
                )
                checks.append(
                    {
                        "method": method,
                        "beta_absolute_tolerance": 2e-4,
                        "probability_relative_tolerance": 1e-3,
                    }
                )
            elif method == "SensitivityAnalysis":
                np.testing.assert_allclose(
                    flatten(record["marginal"]),
                    flatten(previous["marginal"]),
                    rtol=1e-8,
                    atol=1e-9,
                )
                checks.append({"method": method, "absolute_tolerance": 1e-9})
            else:
                uncertainty = np.hypot(
                    record["probability"] * record["cov"],
                    previous["probability"] * previous["cov"],
                )
                standardized_difference = (
                    abs(record["probability"] - previous["probability"]) / uncertainty
                )
                assert standardized_difference <= 5, (method, standardized_difference)
                checks.append(
                    {
                        "method": method,
                        "pooled_standard_errors": float(standardized_difference),
                        "limit": 5,
                    }
                )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="detached checkout or git archive export of v1.6.0",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="skip OpenSees execution, retaining its conversion trial",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in [
        *sorted((args.baseline / "examples").glob("*.py")),
        *sorted((args.baseline / "docs/source/notebooks").glob("*.ipynb")),
    ]:
        before = path.read_text()
        result = (
            convert_notebook(before)
            if path.suffix == ".ipynb"
            else convert_source(before)
        )
        inventory.append(
            {
                "path": str(path.relative_to(args.baseline)),
                "changed": before != result.source,
                "idempotent": True,
                "diagnostics": [asdict(item) for item in result.diagnostics],
            }
        )
    (args.output / "conversion-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    outcomes = []
    for name, original_path in SCRIPTS.items():
        original = args.baseline / original_path
        source = baseline_adjustments(code(original.read_text(), original.suffix), name)
        baseline_script = args.output / (name + "-baseline.py")
        baseline_script.write_text(source)
        converted = convert_source(source).source
        (args.output / (name + "-converted.py")).write_text(converted)
        reviewed = (TRIALS / (name + ".py")).read_text()
        (args.output / (name + "-manual.patch")).write_text(
            "".join(
                difflib.unified_diff(
                    converted.splitlines(True),
                    reviewed.splitlines(True),
                    fromfile=name + " (converter)",
                    tofile=name + " (reviewed)",
                )
            )
        )
        if name == "openseespy_ex" and args.skip_external:
            outcomes.append({"script": original_path, "status": "external_skipped"})
            continue
        executions = {}
        for version, checkout, script in [
            ("1.6.0", args.baseline, baseline_script),
            ("2.0", ROOT, TRIALS / (name + ".py")),
        ]:
            env = {
                **os.environ,
                "PYTHONPATH": str(checkout.resolve() / "src"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(args.output.resolve() / "matplotlib"),
            }
            target = args.output / f"{name}-{version}.json"
            with (args.output / f"{name}-{version}.log").open("w") as log:
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(TRIALS / "observe.py"),
                            str(script),
                            str(target),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    )
                    executions[version] = process.returncode
                except subprocess.TimeoutExpired:
                    executions[version] = "timeout"
        outcome = {"script": original_path, "executions": executions}
        if all(status == 0 for status in executions.values()):
            try:
                outcome["checks"] = compare(
                    json.loads((args.output / f"{name}-1.6.0.json").read_text()),
                    json.loads((args.output / f"{name}-2.0.json").read_text()),
                )
                outcome["status"] = "passed"
            except (AssertionError, IndexError, ValueError) as error:
                outcome.update(status="comparison_failed", error=str(error))
        else:
            outcome["status"] = "execution_failed"
        outcomes.append(outcome)
        print(name, outcome["status"], flush=True)
        (args.output / "results.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    return int(
        any(row["status"] not in ("passed", "external_skipped") for row in outcomes)
    )


if __name__ == "__main__":
    raise SystemExit(main())
