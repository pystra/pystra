"""Execute maintained tutorials in fresh kernels, writing outputs separately."""

import argparse
import os
from pathlib import Path
import re
import time

import nbformat
from nbclient import NotebookClient


def indexed_notebooks(index):
    """Follow explicit local toctrees, including tutorial category pages."""
    notebooks = []
    visited = set()

    def visit(document):
        document = document.resolve()
        if document in visited:
            raise ValueError(f"Repeated tutorial index entry: {document}")
        visited.add(document)
        directive_indent = None
        for line in document.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped == ".. toctree::":
                directive_indent = indent
                continue
            if not stripped:
                continue
            if directive_indent is None:
                continue
            if indent <= directive_indent:
                directive_indent = None
                continue
            if stripped.startswith(":"):
                continue
            # Sphinx also permits an explicit title: Title <relative/path>.
            match = re.fullmatch(r".*<([^<>]+)>", stripped)
            target = match.group(1) if match else stripped
            path = document.parent / target
            notebook = path.with_suffix(".ipynb")
            if notebook.is_file():
                notebook = notebook.resolve()
                if notebook in notebooks:
                    raise ValueError(f"Repeated tutorial notebook: {notebook}")
                notebooks.append(notebook)
            else:
                visit(path.with_suffix(".rst"))

    visit(index)
    return notebooks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*", help="Notebook stems; default: tutorial index"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/source/notebooks"
    names = args.names or [
        path.stem for path in indexed_notebooks(root / "docs/source/tutorial.rst")
    ]
    if not names:
        parser.error("No maintained tutorials found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(root / "src")
    for name in names:
        start = time.monotonic()
        notebook = nbformat.read(source / f"{name}.ipynb", as_version=4)
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(source)}},
        ).execute()
        nbformat.write(notebook, args.output_dir / f"{name}.ipynb")
        print(f"{name}: passed ({time.monotonic() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
