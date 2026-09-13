"""Notebook downloads, figure alternatives and the migration map for the PySTRA docs."""

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
        packages = details.get("packages", [])
        if not all(re.fullmatch(r"[A-Za-z0-9._-]+", name) for name in packages):
            raise ValueError(f"Invalid package name in {path}: {packages}")
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
            f"  python -m pip install {' '.join(['jupyterlab', *packages])}\n\n"
            "The branch changes during development; record the installed commit for\n"
            "a reproducible study. Extract this entire bundle to one directory,\n"
            "launch JupyterLab there, select that environment's Python kernel,\n"
            f"open {path.name}, and run all cells from the top.\n\n"
            f"Optional dependencies: {', '.join([dependencies, *packages])}. "
            "Included helpers: "
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


# Modules of the released 1.6.0 package (tag v1.6.0). Other baseline entries
# came from feature branches merged for 2.0 and were never released in 1.x.
_RELEASED_1X = frozenset(
    """pystra pystra.analysis pystra.calibration pystra.cholesky_sensitivity
    pystra.correlation pystra.distributions pystra.distributions.beta
    pystra.distributions.chisquare pystra.distributions.distribution
    pystra.distributions.gamma pystra.distributions.gev pystra.distributions.gumbel
    pystra.distributions.lognormal pystra.distributions.maximum
    pystra.distributions.normal pystra.distributions.parent
    pystra.distributions.scipydist pystra.distributions.shiftedexponential
    pystra.distributions.shiftedlognormal pystra.distributions.shiftedrayleigh
    pystra.distributions.typeiiismallestvalue pystra.distributions.typeiilargestvalue
    pystra.distributions.typeilargestvalue pystra.distributions.typeismallestvalue
    pystra.distributions.uniform pystra.distributions.weibull
    pystra.distributions.zeroinflated pystra.form pystra.integration pystra.loadcomb
    pystra.ls pystra.mc pystra.model pystra.quadrature pystra.sensitivity pystra.sorm
    pystra.ss pystra.transformation""".split()
)
_CHANGE = {
    "renamed": "renamed",
    "replaced": "replaced",
    "removed": "removed",
    "changed_contract": "changed",
}


def _rst_text(text):
    """Escape inline markup in the manifest's free-text notes."""
    text = text.replace("\\", "\\\\")
    for char in "*|`":
        text = text.replace(char, "\\" + char)
    return re.sub(r"(\w)_(?=\W|$)", r"\1\\_", text)


def _write_migration_map(app):
    """Generate the complete symbol map from the migration manifest."""
    source = Path(app.srcdir)
    manifest = json.loads(
        (source.parent / "migration" / "api-migration.json").read_text()
    )
    modules = set(_RELEASED_1X)
    for entry in manifest["definitions"]:
        parts = entry["old"].split(".")
        for index, part in enumerate(parts[1:], start=1):
            if part[:1].isupper():
                modules.add(".".join(parts[:index]))
                break
    groups = {}
    for entry in manifest["definitions"]:
        parts = entry["old"].split(".")
        if entry["status"] == "unchanged" or any(p.startswith("_") for p in parts):
            continue
        prefixes = [".".join(parts[:i]) for i in range(1, len(parts))]
        module = max((p for p in prefixes if p in modules), key=len, default="pystra")
        groups.setdefault(module, []).append(entry)
    lines = []
    for module in sorted(groups):
        title = f"``{module}``" + ("" if module in _RELEASED_1X else " (not in 1.6.0)")
        lines += [title, "-" * len(title), "", ".. list-table::"]
        lines += ["   :header-rows: 1", "   :widths: 28 32 10 30", ""]
        lines += ["   * - 1.x name", "     - 2.0", "     - Change", "     - Notes"]
        for entry in sorted(groups[module], key=lambda item: item["old"]):
            targets = [entry["current"]] if entry["current"] else []
            targets += entry.get("replacements", [])
            new = ", ".join(f"``{target}``" for target in targets) or "(none)"
            notes = _rst_text(entry.get("notes", ""))
            lines += [f"   * - ``{entry['old'][len(module) + 1:]}``", f"     - {new}"]
            lines += [f"     - {_CHANGE[entry['status']]}"]
            lines += [f"     - {notes}" if notes else "     -"]
        lines.append("")
    target = source / "_generated" / "migration-map.inc"
    target.parent.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    if not target.exists() or target.read_text() != text:
        target.write_text(text)


def setup(app):
    # install.md is a legacy RST source; MyST is used explicitly for the
    # contributor include, without changing the installation page parser.
    app.add_source_suffix(".md", "restructuredtext", override=True)
    app.connect("builder-inited", _write_migration_map)
    app.connect("env-before-read-docs", _prepare_notebooks)
    app.connect("doctree-read", _figure_alternatives)
    app.connect("doctree-read", _notebook_anchors)
    return {"version": "1", "parallel_read_safe": False, "parallel_write_safe": True}
