"""Notebook downloads and meaningful figure alternatives for the PySTRA docs."""

from io import BytesIO
import json
from pathlib import Path
import re
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from docutils import nodes
from sphinx.util import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from execute_notebooks import indexed_notebooks

logger = logging.getLogger(__name__)


def _prepare_notebooks(app, env, docnames):
    """Build deterministic, self-contained bundles before resolving downloads."""
    source = Path(app.srcdir)
    root = source.parents[1]
    destination = source / "_generated/notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    env.pystra_notebooks = {}
    for path in indexed_notebooks(source / "tutorial.rst"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        details = notebook["metadata"]["pystra"]
        dependencies = details["dependencies"]
        helpers = details["support_files"]
        if dependencies not in {"core", "al"}:
            raise ValueError(f"Unknown dependency group in {path}: {dependencies}")
        files = {
            path.name: path.read_bytes(),
            "LICENSE": (root / "LICENSE").read_bytes(),
        }
        for helper in helpers:
            if Path(helper).name != helper or not helper.endswith(".py"):
                raise ValueError(f"Invalid notebook helper: {helper}")
            files[helper] = (path.parent / helper).read_bytes()
        extra = "[al]" if dependencies == "al" else ""
        files["README.txt"] = (
            f"PySTRA {app.config.release}: {path.stem}\n\n"
            "Install the matching development branch in your Python environment:\n\n"
            f'  python -m pip install "pystra{extra} @ '
            'git+https://github.com/pystra/pystra.git@v2.0"\n'
            "  python -m pip install jupyterlab\n\n"
            "The branch changes during development; record the installed commit for\n"
            "a reproducible study. Extract this entire bundle to one directory,\n"
            "launch JupyterLab there, select that environment's Python kernel,\n"
            f"open {path.name}, and run all cells from the top.\n\n"
            f"Optional dependencies: {dependencies}. Included helpers: "
            f"{', '.join(helpers) or 'none'}.\n"
            "Source: https://github.com/pystra/pystra/tree/v2.0/docs/source/notebooks\n"
            "Method sources and benchmark assumptions are cited in the notebook.\n"
        ).encode()
        content = BytesIO()
        with ZipFile(content, "w") as bundle:
            for name, data in sorted(files.items()):
                info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                bundle.writestr(info, data)
        target = destination / f"{path.stem}.zip"
        if not target.exists() or target.read_bytes() != content.getvalue():
            target.write_bytes(content.getvalue())
        env.pystra_notebooks[path.relative_to(source).with_suffix("").as_posix()] = (
            details
        )


def _figure_alternatives(app, doctree):
    """Use per-cell figure descriptions, preserving captions in the notebook."""
    docname = app.env.docname
    source = Path(app.srcdir) / f"{docname}.ipynb"
    if not source.is_file():
        return
    notebook = json.loads(source.read_text(encoding="utf-8"))
    prefix = docname.replace("/", "_")
    seen = {}
    for node in doctree.findall(nodes.image):
        match = re.search(rf"{re.escape(prefix)}_(\d+)_(\d+)\.", node["uri"])
        if match is None:
            continue
        cell_index = int(match[1])
        descriptions = (
            notebook["cells"][cell_index]["metadata"]
            .get("pystra", {})
            .get("figure_alts", [])
        )
        # Output numbers include text; descriptions enumerate image outputs only.
        figure_index = seen.get(cell_index, 0)
        seen[cell_index] = figure_index + 1
        if figure_index >= len(descriptions) or not descriptions[figure_index].strip():
            logger.warning(
                "Missing figure alternative in %s, cell %s",
                docname,
                cell_index,
                location=node,
            )
        else:
            node["alt"] = descriptions[figure_index]


def _notebook_anchors(app, doctree):
    """Retain explicit Markdown anchors that Pandoc otherwise drops."""
    source = Path(app.srcdir) / f"{app.env.docname}.ipynb"
    if not source.is_file():
        return
    notebook = json.loads(source.read_text(encoding="utf-8"))
    titles = {node.astext(): node.parent for node in doctree.findall(nodes.title)}
    known = {
        identifier
        for node in doctree.findall(nodes.Element)
        for identifier in node.get("ids", [])
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        content = "".join(cell["source"])
        for alias, heading in re.findall(
            r'<a id="([^"]+)"></a>\s*#{1,6} ([^\n]+)', content
        ):
            if alias in known:
                continue
            section = titles.get(heading)
            if section is None:
                logger.warning(
                    "Cannot retain notebook anchor %s in %s", alias, app.env.docname
                )
                continue
            section.parent.insert(
                section.parent.index(section), nodes.target(ids=[alias])
            )
            known.add(alias)


def setup(app):
    # install.md is a legacy RST source; MyST is used explicitly for the
    # contributor include, without changing the installation page parser.
    app.add_source_suffix(".md", "restructuredtext", override=True)
    app.connect("env-before-read-docs", _prepare_notebooks)
    app.connect("doctree-read", _figure_alternatives)
    app.connect("doctree-read", _notebook_anchors)
    return {"version": "1", "parallel_read_safe": False, "parallel_write_safe": True}
