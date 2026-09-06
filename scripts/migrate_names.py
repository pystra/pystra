"""Apply the reviewed first-pass naming map to tracked repository files.

This is a repository maintenance tool, not a general user-code codemod: it
does not infer receiver types. Dry-run is the default. Numeric defaults,
parameter names, user variable names, and notebook outputs are preserved.
"""

import argparse
import ast
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize


def substitute(text, mapping):
    pattern = r"\b(?:" + "|".join(map(re.escape, mapping)) + r")\b"
    return re.sub(pattern, lambda match: mapping[match.group()], text)


def prose(text, classes, callables):
    text = substitute(text, callables)
    for old, new in classes.items():
        if old not in ("DDO", "LQI", "SWTP"):
            text = re.sub(rf"\b{old}\b", new, text)
            continue
        # References and code-like expressions, preserving acronyms in prose.
        text = re.sub(rf"(?<=[.`~]){old}\b", new, text)
        text = re.sub(rf"\b{old}(?=\s*\()", new, text)
        text = re.sub(rf"^(\s+){old}(\s*)$", rf"\g<1>{new}\g<2>", text, flags=re.M)
    return text


def python_source(text, classes, callables):
    mapping = {**classes, **callables}
    attribute_strings = set()
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def mark_strings(node):
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                column = len(
                    lines[child.lineno - 1].encode()[: child.col_offset].decode()
                )
                attribute_strings.add((child.lineno, column))

    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Notebook cells may contain IPython syntax. Name tokens remain usable.
        tree = ast.Module(body=[], type_ignores=[])
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            mark_strings(node.value)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if (
                name in ("getattr", "setattr", "hasattr", "delattr")
                and len(node.args) > 1
            ):
                mark_strings(node.args[1])
        elif isinstance(node, ast.arg) and node.annotation is not None:
            mark_strings(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            mark_strings(node.annotation)

    edits = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        replacement = token.string
        if token.type == tokenize.NAME:
            replacement = mapping.get(token.string, token.string)
            if token.string.startswith("test_"):
                replacement = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", replacement)
                replacement = re.sub(
                    r"([a-z0-9])([A-Z])", r"\1_\2", replacement
                ).lower()
        elif token.type == tokenize.COMMENT:
            replacement = prose(token.string, classes, callables)
        elif token.type == tokenize.STRING:
            replacement = prose(token.string, classes, callables)
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, str) and (
                token.start in attribute_strings or value in callables
            ):
                replacement = substitute(replacement, mapping)
        if replacement != token.string:
            start = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            edits.append((start, end, replacement))
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    mapping = json.loads((root / "docs/migration/naming-map.json").read_text())
    classes, callables = mapping["classes"], mapping["callables"]
    paths = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    changed = []
    for name in filter(None, paths):
        path = root / name
        if not name.startswith(("src/", "tests/", "examples/", "docs/source/")):
            continue
        if path.suffix not in (".py", ".rst", ".md", ".ipynb"):
            continue
        before = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            after = python_source(before, classes, callables)
            ast.parse(after)
        elif path.suffix == ".ipynb":
            notebook = json.loads(before)
            for cell in notebook.get("cells", []):
                source = cell.get("source", [])
                text = "".join(source) if isinstance(source, list) else source
                transform = python_source if cell["cell_type"] == "code" else prose
                text = transform(text, classes, callables)
                cell["source"] = (
                    text.splitlines(keepends=True) if isinstance(source, list) else text
                )
            after = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        elif name == "docs/source/changelog.rst":
            current, separator, history = before.partition("v1.6.0 (2026-03-16)")
            after = prose(current, classes, callables) + separator + history
        else:
            after = prose(before, classes, callables)
        if after != before:
            changed.append(name)
            if args.write:
                path.write_text(after, encoding="utf-8")
    print("\n".join(changed))
    print(
        f"{'Updated' if args.write else 'Would update'} {len(changed)} tracked files."
    )


if __name__ == "__main__":
    main()
