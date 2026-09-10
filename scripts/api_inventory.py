"""Inventory PySTRA definitions and observed exports without running analyses.

This captures the existing surface, including incidental exports and private
helpers. An observed symbol is not automatically a supported extension point.
"""

import argparse
import ast
from collections import Counter
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys


def definitions(root):
    entries = []
    for path in sorted((root / "src/pystra").rglob("*.py")):
        relative = path.relative_to(root / "src").with_suffix("")
        module = ".".join(relative.parts).removesuffix(".__init__")
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(body, prefix):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    qualified_name = f"{prefix}.{node.name}"
                    attributes = {
                        item.attr
                        for item in ast.walk(node)
                        if isinstance(item, ast.Attribute)
                        and isinstance(item.ctx, ast.Store)
                        and isinstance(item.value, ast.Name)
                        and item.value.id == "self"
                    }
                    fields = {
                        item.target.id
                        for item in node.body
                        if isinstance(item, ast.AnnAssign)
                        and isinstance(item.target, ast.Name)
                    }
                    entries.append(
                        {
                            "name": qualified_name,
                            "kind": "class",
                            "bases": [ast.unparse(base) for base in node.bases],
                            "attributes": sorted(attributes | fields),
                        }
                    )
                    visit(node.body, qualified_name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entries.append(
                        {
                            "name": f"{prefix}.{node.name}",
                            "kind": "callable",
                            "signature": f"({ast.unparse(node.args)})",
                            "decorators": [ast.unparse(d) for d in node.decorator_list],
                            "returns": (
                                ast.unparse(node.returns) if node.returns else None
                            ),
                        }
                    )

        visit(tree.body, module)
    return entries


def check_names(root):
    errors = []
    mapping_path = root / "docs/migration/naming-map.json"
    legacy_classes = (
        json.loads(mapping_path.read_text())["classes"] if mapping_path.exists() else {}
    )
    for path in sorted((root / "src/pystra").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                c.isupper() for c in node.name
            ):
                errors.append(f"{path.relative_to(root)}:{node.lineno}: {node.name}")
            elif isinstance(node, ast.ClassDef) and node.name in legacy_classes:
                errors.append(f"{path.relative_to(root)}:{node.lineno}: {node.name}")
    if errors:
        raise SystemExit("Naming violations:\n" + "\n".join(errors))
    print("Function, method, and migrated class naming check passed.")


def inventory(root):
    sys.path.insert(0, str(root / "src"))
    package = importlib.import_module("pystra")
    exports = {}
    for module_name in ("pystra", "pystra.distributions", "pystra.decision.ddo"):
        module = importlib.import_module(module_name)
        exports[module_name] = {
            name: f"{obj.__module__}.{obj.__qualname__}"
            for name, obj in sorted(vars(module).items())
            if not name.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
        }
    return {
        "schema_version": 1,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "package_version": package.__version__,
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "matplotlib", "pytest")
        },
        "definitions": definitions(root),
        "observed_exports": exports,
    }


def check_migration(root):
    """Check baseline coverage and references in the reviewed migration records."""
    directory = root / "docs/migration"
    baseline = json.loads((directory / "api-baseline.json").read_text())
    migration = json.loads((directory / "api-migration.json").read_text())
    structure = json.loads((directory / "calibration-structure-map.json").read_text())
    current = definitions(root)
    names = {d["name"] for d in current}
    errors = []
    if Counter(d["name"] for d in baseline["definitions"]) != Counter(
        d["old"] for d in migration["definitions"]
    ):
        errors.append("Migration manifest must retain every baseline definition")
    for entry in migration["definitions"]:
        references = entry.get("replacements", [])
        if entry["current"] is not None:
            references = [entry["current"], *references]
        elif entry["status"] not in ("removed", "replaced"):
            errors.append(f"Missing disposition for {entry['old']}")
        if entry["status"] == "replaced" and not entry.get("replacements"):
            errors.append(f"Missing replacement for {entry['old']}")
        for name in references:
            if name not in names:
                errors.append(f"Unknown definition {name} for {entry['old']}")
    for workflow in structure["replacement_workflows"].values():
        for name in workflow:
            if name not in names:
                errors.append(f"Unknown workflow definition {name}")
    expected = [
        d
        for d in current
        if d["name"].startswith(
            ("pystra.calibration.", "pystra.loadcomb.", "pystra.results.")
        )
        or d["name"] == "pystra.reliability.form.FORM.run"
    ]
    if structure["definitions"] != expected:
        errors.append("Calibration structure inventory differs from current source")
    if errors:
        raise SystemExit("Migration errors:\n" + "\n".join(errors))
    print(
        "Migration coverage, definition references, and calibration contracts passed."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-names", action="store_true")
    parser.add_argument("--check-migration", action="store_true")
    args = parser.parse_args()
    if args.check_names:
        check_names(args.root)
    if args.check_migration:
        check_migration(args.root)
    if not (args.check_names or args.check_migration):
        if args.output is None:
            parser.error("--output is required when capturing an inventory")
        data = inventory(args.root.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Captured {len(data['definitions'])} definitions in {args.output}.")


if __name__ == "__main__":
    main()
