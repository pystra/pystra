"""The curated top-level namespace and the signposts for names that moved in 2.0."""

import importlib
import json
import pathlib
import re
import subprocess
import sys

import pytest

import pystra
import pystra.distributions
from pystra import _signposts

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = json.loads((ROOT / "docs/migration/module-map.json").read_text())["modules"]
STUBS = [m for m in MODULES if m["released"] and m["old"].count(".") == 1]

TOP_LEVEL = {
    # models and distributions
    "StochasticModel",
    "LimitState",
    "Constant",
    "CorrelationMatrix",
    "Distribution",
    "StdNormal",
    "Normal",
    "Lognormal",
    "Uniform",
    "Beta",
    "Gamma",
    "ChiSquare",
    "ShiftedExponential",
    "ShiftedLognormal",
    "ShiftedRayleigh",
    "Gumbel",
    "GumbelMin",
    "Frechet",
    "Weibull",
    "GEV",
    "GEVmax",
    "GEVMin",
    "Maximum",
    "MaxParent",
    "ZeroInflated",
    "ScipyDist",
    # dependence
    "JointDistribution",
    "GaussianCopula",
    "StudentTCopula",
    "FrankCopula",
    "IndependentCopula",
    "Transformation",
    # reliability methods and results
    "AnalysisObject",
    "AnalysisOptions",
    "FORM",
    "SORM",
    "MonteCarlo",
    "CrudeMonteCarlo",
    "ImportanceSampling",
    "DistributionAnalysis",
    "LineSampling",
    "SubsetSimulation",
    "SensitivityAnalysis",
    "SystemFORM",
    "StrongMaximumTest",
    "ActiveLearning",
    "ActiveLearningResult",
    "FORMResult",
    # systems and loads
    "Component",
    "System",
    "SeriesSystem",
    "ParallelSystem",
    "CutSetSystem",
    "TieSetSystem",
    "KOfNSystem",
    "ditlevsen_bounds",
    "FBCProcess",
    "VariableRoles",
    "LoadCombination",
}


def test_top_level_namespace_is_curated():
    assert set(pystra.__all__) == TOP_LEVEL
    assert len(pystra.__all__) == len(TOP_LEVEL)
    assert all(name in vars(pystra) for name in pystra.__all__)


@pytest.mark.parametrize(
    "name", ["np", "optimize", "OrderedDict", "normal", "scipy_norm"]
)
def test_incidental_names_do_not_leak(name):
    assert name not in vars(pystra)


@pytest.mark.parametrize("name, message", sorted(_signposts.TOP_LEVEL.items()))
def test_top_level_signposts_name_the_replacement(name, message):
    with pytest.raises(AttributeError) as info:
        getattr(pystra, name)
    assert str(info.value) == message
    assert not hasattr(pystra, name)


@pytest.mark.parametrize("name, message", sorted(_signposts.DISTRIBUTIONS.items()))
def test_distribution_signposts_name_the_replacement(name, message):
    with pytest.raises(AttributeError) as info:
        getattr(pystra.distributions, name)
    assert str(info.value) == message


@pytest.mark.parametrize("module", STUBS, ids=[m["old"] for m in STUBS])
def test_released_module_paths_name_their_new_home(module):
    with pytest.raises(ImportError, match=re.escape(module["new"])):
        importlib.import_module(module["old"])


def test_subpackage_names_point_to_their_subpackage():
    with pytest.raises(AttributeError, match=r"import it from pystra\.calibration"):
        pystra.CodeCalibration


def test_unknown_names_raise_the_usual_error():
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        pystra.nonexistent


def test_signposts_are_current_with_the_migration_records():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_signposts.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "subpackage, name",
    [
        ("reliability", "FORM"),
        ("reliability", "ImportanceSampling"),
        ("dependence", "Transformation"),
        ("loads", "LoadCombination"),
        ("active_learning", "ActiveLearning"),
    ],
)
def test_subpackages_export_the_top_level_objects(subpackage, name):
    module = importlib.import_module(f"pystra.{subpackage}")
    assert getattr(module, name) is getattr(pystra, name)


def test_decision_tools_live_in_pystra_decision():
    assert pystra.decision.DDO.__module__ == "pystra.decision.ddo"
