"""Naming checks preserve the AST visitor protocol without masking violations."""

import importlib.util
from pathlib import Path

import pytest


def _check_names(tmp_path, source):
    specification = importlib.util.spec_from_file_location(
        "api_inventory",
        Path(__file__).resolve().parents[1] / "scripts/api_inventory.py",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    package = tmp_path / "src/pystra"
    package.mkdir(parents=True)
    (package / "visitor.py").write_text(source, encoding="utf-8")
    module.check_names(tmp_path)


@pytest.mark.parametrize("base", ["ast.NodeVisitor", "ast.NodeTransformer"])
def test_ast_visitor_protocol_names_are_allowed(tmp_path, base):
    _check_names(
        tmp_path,
        f"class Visitor({base}):\n    def visit_FunctionDef(self, node): pass\n",
    )


@pytest.mark.parametrize(
    "source, name",
    [
        ("class Other:\n    def visit_Name(self, node): pass\n", "visit_Name"),
        (
            "class Visitor(ast.NodeVisitor):\n"
            "    def visit_Typo(self, node): pass\n",
            "visit_Typo",
        ),
        (
            "class Visitor(ast.NodeVisitor):\n"
            "    def visit_Name(self, node):\n"
            "        def visit_Call(): pass\n",
            "visit_Call",
        ),
        (
            "class Visitor(ast.NodeVisitor):\n"
            "    def processNode(self, node): pass\n",
            "processNode",
        ),
    ],
)
def test_other_mixed_case_names_are_rejected(tmp_path, source, name):
    with pytest.raises(SystemExit, match=name):
        _check_names(tmp_path, source)
