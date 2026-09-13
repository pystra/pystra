"""Preview or apply conservative PySTRA 1.x script migration."""

import argparse
import difflib
from pathlib import Path
import sys
import tokenize

from . import convert_notebook, convert_source


def main(argv: list[str] | None = None) -> int:
    """Emit diffs and diagnostics; return 1 for review items, 2 for errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Python files, notebooks, or directories"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="apply changes (default: show unified diffs)",
    )
    args = parser.parse_args(argv)
    paths = set()
    for path in args.paths:
        if path.is_dir():
            paths.update(
                p
                for p in path.rglob("*")
                if p.suffix in (".py", ".ipynb")
                and not any(part.startswith(".") for part in p.relative_to(path).parts)
            )
        else:
            paths.add(path)
    status = 0
    for path in sorted(paths):
        try:
            if path.suffix == ".py":
                with path.open("rb") as stream:
                    encoding, _ = tokenize.detect_encoding(stream.readline)
                before = path.read_bytes().decode(encoding)
                result = convert_source(before)
            elif path.suffix == ".ipynb":
                encoding = "utf-8"
                before = path.read_bytes().decode(encoding)
                result = convert_notebook(before)
            else:
                raise ValueError("expected a .py or .ipynb file")
            for diagnostic in result.diagnostics:
                cell = f":cell {diagnostic.cell}" if diagnostic.cell else ""
                print(
                    f"{path}{cell}:{diagnostic.line}:{diagnostic.column}: review: {diagnostic.message}",
                    file=sys.stderr,
                )
                status = max(status, 1)
            if before != result.source:
                sys.stdout.writelines(
                    difflib.unified_diff(
                        before.splitlines(keepends=True),
                        result.source.splitlines(keepends=True),
                        fromfile=str(path),
                        tofile=str(path) + " (2.0)",
                    )
                )
                if args.write:
                    path.write_bytes(result.source.encode(encoding))
        except (OSError, ValueError, SyntaxError, KeyError) as error:
            print(f"{path}: error: {error}", file=sys.stderr)
            status = 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
