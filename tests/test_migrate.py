"""The user converter changes only bindings whose PySTRA origin is known."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pystra.migrate import convert_notebook, convert_source
from pystra.migrate.__main__ import main


@pytest.mark.parametrize(
    "before, after",
    [
        (
            "import pystra as ra\nx = ra.Form(stochastic_model=m, analysis_options=o, limit_state=g)\n",
            "import pystra as ra\nx = ra.FORM(model=m, options=o, limit_state=g)\n",
        ),
        (
            "from pystra import Form\nx = Form(stochastic_model=m, limit_state=g)\n",
            "from pystra import FORM as Form\nx = Form(model=m, limit_state=g)\n",
        ),
        (
            "from pystra.form import Form as Solver\nx = Solver(stochastic_model=m)\n",
            "from pystra.reliability.form import FORM as Solver\nx = Solver(model=m)\n",
        ),
        (
            "import pystra.form as fm\nx = fm.Form(stochastic_model=m)\n",
            "import pystra.reliability.form as fm\nx = fm.FORM(model=m)\n",
        ),
        (
            "import pystra.form\nx = pystra.form.Form(stochastic_model=m)\n",
            "import pystra.reliability.form\nx = pystra.reliability.form.FORM(model=m)\n",
        ),
        (
            "from pystra import form\nx = form.Form(stochastic_model=m)\n",
            "from pystra.reliability import form\nx = form.FORM(model=m)\n",
        ),
        (
            "from pystra.distributions import TypeIlargestValue\nx = TypeIlargestValue('X', mean=3, stdv=1, startpoint=2)\n",
            "from pystra.distributions import Gumbel as TypeIlargestValue\nx = TypeIlargestValue('X', mean=3, std=1, start_point=2)\n",
        ),
        (
            "import pystra as ra\nx = ra.Normal(  'stdv', mean=3, stdv = 1 ) # Form\n",
            "import pystra as ra\nx = ra.Normal(  'stdv', mean=3, std = 1 ) # Form\n",
        ),
        (
            "import pystra as ra\ns = 'é'; x = ra.Form(stochastic_model=m)\n",
            "import pystra as ra\ns = 'é'; x = ra.FORM(model=m)\n",
        ),
        (
            "from pystra.mc import ImportanceSampling\nx = ImportanceSampling(stochastic_model=m)\n",
            "from pystra.reliability.importance_sampling import ImportanceSampling\nx = ImportanceSampling(model=m)\n",
        ),
    ],
)
def test_resolved_edits(before, after):
    result = convert_source(before)
    assert result.source == after
    assert convert_source(result.source).source == after


@pytest.mark.parametrize(
    "source",
    [
        "def Form(stdv):\n    return stdv\nx = Form(stdv=3)\n",
        "from other import Form\nx = Form(stdv=3)\n",
        "import pystra as ra\nra = user_object\nx = ra.Form(stdv=3)\n",
        "import pystra as ra\ndef f(ra):\n    return ra.Form(stdv=3)\n",
        "import pystra as ra\nf = lambda ra: ra.Form(stdv=3)\n",
        "import pystra as ra\nx = [ra.Form(stdv=3) for ra in values]\n",
        "import pystra as ra\ndef f():\n    ra = user_object\n    return ra.Form(stdv=3)\n",
        "import pystra as ra\ntry:\n    work()\nexcept Exception as ra:\n    ra.Form(stdv=3)\n",
        "import pystra as ra\nmatch obj:\n    case {'value': ra}:\n        ra.Form(stdv=3)\n",
        "import pystra as ra\ns = 'ra.Form(stdv=3)'\ns = f'{ra.Form}'\n# ra.Form(stdv=3)\n",
        "import pystra as ra\ndef f():\n    global ra\n    ra = user_object\nx = ra.Form(stdv=3)\n",
        "from pystra import Form as solver\nsolver = custom\nx = solver(stdv=3)\n",
    ],
)
def test_user_names_and_shadowed_bindings_are_untouched(source):
    result = convert_source(source)
    # The import itself may safely rename its exported symbol, preserving alias.
    assert result.source.replace("FORM as solver", "Form as solver") == source


def test_shadowing_does_not_hide_independent_module_scope():
    source = "import pystra as ra\ndef f(ra):\n    return ra.Form(stdv=3)\nx = ra.Form(stochastic_model=m)\n"
    result = convert_source(source)
    assert "return ra.Form(stdv=3)" in result.source
    assert "x = ra.FORM(model=m)" in result.source


@pytest.mark.parametrize(
    "code, expected",
    [
        ("opts = ra.AnalysisOptions()", "manual migration"),
        ("x = ra.Normal('X', 3, 1, input_type='par')", "input_type"),
        ("form.getBeta()", "getBeta"),
        ("form.showDetailedOutput()", "showDetailedOutput"),
        ("c = ra.Calibration()", "manual migration"),
        ("x = ra.Form(**settings)", "Expanded constructor"),
        ("x = ra.Form(stochastic_model=m, model=n)", "duplicate"),
        (
            "import numpy as np\nnp.random.seed(42)\nx = ra.CrudeMonteCarlo()",
            "Global NumPy seeding",
        ),
    ],
)
def test_manual_review_diagnostics(code, expected):
    result = convert_source("import pystra as ra\n" + code + "\n")
    assert any(expected in diagnostic.message for diagnostic in result.diagnostics)
    if "input_type" in code:
        assert "input_type='par'" in result.source
    if "getBeta" in code:
        assert "form.getBeta()" in result.source


def test_mixed_module_import_is_flagged_and_unchanged():
    source = "from pystra.mc import CrudeMonteCarlo, ImportanceSampling\n"
    result = convert_source(source)
    assert result.source == source
    assert any("split this import" in d.message for d in result.diagnostics)


def test_wildcard_and_invalid_syntax_are_unchanged():
    for source in (
        "from pystra import *\nx = Form(stochastic_model=m)",
        "%matplotlib inline\nimport pystra as ra\nx = ra.Form()",
    ):
        result = convert_source(source)
        assert result.source == source
        assert result.diagnostics


def test_notebook_carries_imports_and_preserves_noncode_json():
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["ra.Form(stdv=3)"]},
            {
                "cell_type": "code",
                "source": ["import pystra as ra\n"],
                "outputs": [{"text": "Form"}],
            },
            {
                "cell_type": "code",
                "source": ["x = ra.Normal('stdv', 3, stdv=1)\n", "x.getMean()\n"],
                "outputs": [],
            },
        ],
        "metadata": {"Form": "stdv"},
    }
    source = json.dumps(notebook, indent=2)
    result = convert_notebook(source)
    converted = json.loads(result.source)
    assert converted["cells"][2]["source"][0] == "x = ra.Normal('stdv', 3, std=1)\n"
    assert converted["cells"][:2] == notebook["cells"][:2]
    assert converted["metadata"] == notebook["metadata"]
    assert '"Form": "stdv"' in result.source
    assert result.diagnostics[0].cell == 3
    assert convert_notebook(result.source).source == result.source


def test_cli_dry_run_and_explicit_write(tmp_path, capsys):
    path = tmp_path / "user code.py"
    original = b"# coding: latin-1\r\nimport pystra as ra\r\nx = ra.Form() # \xe9\r\n"
    path.write_bytes(original)
    assert main([str(path)]) == 0
    assert path.read_bytes() == original
    assert "+x = ra.FORM()" in capsys.readouterr().out
    assert main(["--write", str(path)]) == 0
    assert path.read_bytes() == original.replace(b"ra.Form()", b"ra.FORM()")
    assert main(["--write", str(path)]) == 0
    assert main([str(tmp_path / "missing.py")]) == 2


def test_installed_command_is_available():
    result = subprocess.run(
        [sys.executable, "-m", "pystra.migrate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--write" in result.stdout


def test_packaged_mapping_agrees_with_manifest():
    path = Path(__file__).resolve().parents[1] / "scripts/generate_migration_data.py"
    spec = importlib.util.spec_from_file_location("generate_migration_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        path.parents[1] / "src/pystra/migrate/_data.py"
    ).read_text() == module.generate()


def test_annotations_use_the_enclosing_import_scope():
    source = (
        "import pystra as ra\ndef f(x: ra.Form) -> ra.Form:\n    return ra.Form()\n"
    )
    assert convert_source(source).source.count("ra.FORM") == 3


def test_walrus_in_comprehension_does_not_misresolve_outer_alias():
    source = "import pystra as ra\nx = [(ra := value) for value in values]\nra.Form(stdv=1)\n"
    assert convert_source(source).source == source


def test_mutated_module_member_is_not_renamed():
    source = "import pystra as ra\nra.Form = user_solver\nra.Form(stdv=1)\n"
    assert convert_source(source).source == source


def test_module_alias_with_displaced_members_is_flagged():
    source = "import pystra.mc as mc\nx = mc.ImportanceSampling()\n"
    result = convert_source(source)
    assert result.source == source
    assert any("separate modules" in d.message for d in result.diagnostics)


def test_repeated_identical_imports_remain_resolvable():
    source = "import pystra as ra\nimport pystra as ra\nx = ra.Form()\n"
    assert "ra.FORM()" in convert_source(source).source


def test_positional_configuration_is_flagged():
    source = "import pystra as ra\nx = ra.Uniform('X', 1, 2, 1)\n"
    result = convert_source(source)
    assert result.source == source
    assert any("Positional configuration" in d.message for d in result.diagnostics)


def test_notebook_flags_seed_before_later_simulation():
    source = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "import numpy as np\n",
                        "import pystra as ra\n",
                        "np.random.seed(3)\n",
                    ],
                },
                {"cell_type": "code", "source": ["s = ra.CrudeMonteCarlo()\n"]},
            ]
        }
    )
    result = convert_notebook(source)
    assert any(
        d.cell == 1 and d.line == 3 and "Global NumPy" in d.message
        for d in result.diagnostics
    )


def test_unaliased_module_with_displaced_members_is_flagged():
    source = "import pystra.mc\nx = pystra.mc.ImportanceSampling()\n"
    result = convert_source(source)
    assert result.source == source
    assert any("separate modules" in d.message for d in result.diagnostics)


def test_private_relocated_module_requires_review():
    for source in [
        "import pystra.integration\n",
        "from pystra.integration import quadrature\n",
    ]:
        result = convert_source(source)
        assert result.source == source
        assert any("Private PySTRA module" in d.message for d in result.diagnostics)


def test_class_moved_out_of_top_level_uses_public_subpackage():
    source = "import pystra as ra\nx = ra.DDO()\n"
    assert (
        convert_source(source).source == "import pystra as ra\nx = ra.decision.DDO()\n"
    )
    source = "from pystra import DDO\nx = DDO()\n"
    assert (
        convert_source(source).source == "from pystra.decision import DDO\nx = DDO()\n"
    )
