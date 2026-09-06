"""Execute maintained tutorials in fresh kernels, writing outputs separately."""

import argparse
import os
from pathlib import Path
import re
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*", help="Notebook stems; default: tutorial index"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/source/notebooks"
    names = args.names or re.findall(
        r"^\s+notebooks/(\w+)\s*$",
        (root / "docs/source/tutorial.rst").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
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
