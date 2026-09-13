"""AST resolution with source-span edits; source is never unparsed."""

import ast
from dataclasses import dataclass, field
import json
import re

from ._data import DATA


@dataclass(frozen=True)
class Diagnostic:
    """A manual-review location (one-based line and column, optional cell)."""

    line: int
    column: int
    message: str
    cell: int | None = None


@dataclass(frozen=True)
class Conversion:
    """Converted source text and unresolved migration diagnostics."""

    source: str
    diagnostics: tuple[Diagnostic, ...]


@dataclass
class _Scope:
    parent: "_Scope | None" = None
    bindings: dict[str, list[str | None]] = field(default_factory=dict)
    star: bool = False

    def bind(self, name, path=None):
        self.bindings.setdefault(name, []).append(path)

    def resolve(self, name):
        if name in self.bindings:
            paths = self.bindings[name]
            return (
                paths[0]
                if paths[0] is not None and all(path == paths[0] for path in paths)
                else None
            )
        if self.star:
            return None
        return self.parent.resolve(name) if self.parent else None


class _Bindings(ast.NodeVisitor):
    """Collect lexical bindings before resolving uses, including later stores."""

    def __init__(self, tree):
        self.current = _Scope()
        self.scopes = {}
        self.visit(tree)

    def visit(self, node):
        self.scopes[id(node)] = self.current
        return super().visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.current.bind(node.id)

    def visit_Import(self, node):
        for item in node.names:
            self.current.bind(
                item.asname or item.name.split(".")[0],
                item.name if item.asname else item.name.split(".")[0],
            )

    def visit_ImportFrom(self, node):
        for item in node.names:
            if item.name == "*":
                self.current.star = True
            else:
                self.current.bind(
                    item.asname or item.name,
                    f"{node.module}.{item.name}" if not node.level else None,
                )

    def visit_FunctionDef(self, node):
        self.current.bind(node.name)
        for expression in [
            *node.decorator_list,
            *node.args.defaults,
            *node.args.kw_defaults,
            node.returns,
        ]:
            if expression is not None:
                self.visit(expression)
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ]:
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)
        parent = self.current
        self.current = _Scope(parent)
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ]:
            if argument is not None:
                self.current.bind(argument.arg)
        for statement in node.body:
            self.visit(statement)
        self.current = parent

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        parent = self.current
        for expression in [*node.args.defaults, *node.args.kw_defaults]:
            if expression is not None:
                self.visit(expression)
        self.current = _Scope(parent)
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ]:
            if argument is not None:
                self.current.bind(argument.arg)
        self.visit(node.body)
        self.current = parent

    def visit_ClassDef(self, node):
        self.current.bind(node.name)
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        parent = self.current
        self.current = _Scope(parent)
        # Conservatively shadow class bindings in methods as well.
        for statement in node.body:
            self.visit(statement)
        self.current = parent

    def visit_Global(self, node):
        # Mutating a global/nonlocal import makes its uses ambiguous.
        scope = self.current
        while scope:
            for name in node.names:
                scope.bind(name)
            scope = scope.parent

    visit_Nonlocal = visit_Global

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            # Comprehension assignment expressions can bind an outer scope.
            scope = self.current
            while scope:
                scope.bind(node.target.id)
                scope = scope.parent
        self.visit(node.target)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                scope = self.current
                while scope:
                    scope.bind(root.id)
                    scope = scope.parent
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.current.bind(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.current.bind(node.name)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest:
            self.current.bind(node.rest)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        parent = self.current
        self.visit(node.generators[0].iter)
        self.current = _Scope(parent)
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for key in ("elt", "key", "value"):
            if hasattr(node, key):
                self.visit(getattr(node, key))
        self.current = parent

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


def _module(path):
    for old, new in sorted(DATA["modules"].items(), key=lambda pair: -len(pair[0])):
        if path == old or path.startswith(old + "."):
            return new + path[len(old) :]
    return path


def _symbol(path):
    return DATA["symbols"].get(path)


class _Editor(ast.NodeVisitor):
    def __init__(self, source, tree):
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.offsets = [0]
        for line in self.lines:
            self.offsets.append(self.offsets[-1] + len(line))
        self.bindings = _Bindings(tree)
        self.edits = {}
        self.diagnostics = []
        self.has_pystra = any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(
                (
                    item.name if isinstance(node, ast.Import) else node.module or ""
                ).split(".")[0]
                == "pystra"
                for item in node.names
            )
            for node in ast.walk(tree)
        )
        self.simulation = False
        self.seeds = []
        self.split_modules = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            path = self.resolve(node)
            record = _symbol(path) if path else None
            if record and record["target"]:
                bound = path.rsplit(".", 1)[0]
                if (
                    bound not in ("pystra", "pystra.distributions")
                    and bound in DATA["modules"]
                    and record["target"].rsplit(".", 1)[0] != _module(bound)
                ):
                    self.split_modules.add(bound)

    def offset(self, line, byte_column):
        return self.offsets[line - 1] + len(
            self.lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8")
        )

    def span(self, node):
        return self.offset(node.lineno, node.col_offset), self.offset(
            node.end_lineno, node.end_col_offset
        )

    def edit(self, start, end, value):
        if self.source[start:end] != value:
            self.edits[start, end] = value

    def warn(self, node, message):
        self.diagnostics.append(
            Diagnostic(
                node.lineno,
                len(self.lines[node.lineno - 1].encode()[: node.col_offset].decode())
                + 1,
                message,
            )
        )

    def resolve(self, node):
        if isinstance(node, ast.Name):
            return self.bindings.scopes[id(node)].resolve(node.id)
        if isinstance(node, ast.Attribute):
            prefix = self.resolve(node.value)
            return prefix + "." + node.attr if prefix else None
        return None

    def visit_JoinedStr(self, node):
        # Leave string literals, including embedded formatted expressions, intact.
        return

    def visit_Import(self, node):
        for item in node.names:
            if item.name in self.split_modules:
                self.warn(
                    node,
                    "This module's members moved to separate modules; replace its imports manually.",
                )
                continue
            new = _module(item.name)
            if new != item.name:
                start, _ = self.span(item)
                self.edit(start, start + len(item.name), new)
            if item.name.startswith("pystra._") or item.name in DATA["private_modules"]:
                self.warn(node, "Private PySTRA module requires manual review.")

    def visit_ImportFrom(self, node):
        if node.level or not node.module or not node.module.startswith("pystra"):
            return
        if any(item.name == "*" for item in node.names):
            self.warn(
                node, "Wildcard PySTRA import: use explicit imports before conversion."
            )
            return
        if node.module in DATA["private_modules"]:
            self.warn(node, "Private PySTRA module requires manual review.")
            return
        new_module = _module(node.module)
        imports = []
        destinations = set()
        for item in node.names:
            path = node.module + "." + item.name
            if path in self.split_modules:
                self.warn(
                    node,
                    "This module's members moved to separate modules; replace its imports manually.",
                )
                return
            record = _symbol(path)
            if record:
                if record["manual"]:
                    self.warn(
                        node,
                        f"{item.name} needs manual migration of its options, inputs or workflow.",
                    )
                    destinations.add(node.module)
                    imports.append((item, item.name))
                    continue
                target_module, name = record["target"].rsplit(".", 1)
                destination = target_module
                if node.module in ("pystra", "pystra.distributions"):
                    public_parent, public_name = record["public_path"].rsplit(".", 1)
                    if (
                        node.module == "pystra.distributions"
                        and public_parent == "pystra"
                    ):
                        public_parent = node.module
                    destination, name = public_parent, public_name
            elif path in DATA["modules"]:
                destination, name = DATA["modules"][path].rsplit(".", 1)
            else:
                destination, name = new_module, item.name
            destinations.add(destination)
            imports.append((item, name))
        if len(destinations) != 1:
            self.warn(
                node,
                "Imported objects moved to different modules; split this import manually.",
            )
            return
        destination = destinations.pop()
        if destination != node.module:
            start, end = self.span(node)
            match = re.search(r"\bfrom\s+([^\s]+)\s+import\b", self.source[start:end])
            if match:
                self.edit(start + match.start(1), start + match.end(1), destination)
        for item, name in imports:
            if name != item.name:
                start, _ = self.span(item)
                # Keep every locally bound identifier and user alias unchanged.
                replacement = name if item.asname else f"{name} as {item.name}"
                self.edit(start, start + len(item.name), replacement)

    def visit_Attribute(self, node):
        path = self.resolve(node)
        if path and path.startswith("pystra."):
            record = _symbol(path)
            if record:
                if record["manual"]:
                    self.warn(
                        node,
                        f"{node.attr} needs manual migration of its options, inputs or workflow.",
                    )
                else:
                    name = record["target"].rsplit(".", 1)[1]
                    parent = self.resolve(node.value)
                    if parent == "pystra":
                        name = record["public_path"][len("pystra.") :]
                    _, end = self.span(node)
                    self.edit(end - len(node.attr), end, name)
            # Relocate the longest module expression only, preserving its alias.
            if path in DATA["modules"] and path not in self.split_modules:
                start, end = self.span(node)
                root = node
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    bound = self.resolve(root)
                    target = DATA["modules"][path]
                    bound_new = _module(bound)
                    if target.startswith(bound_new + "."):
                        self.edit(start, end, root.id + target[len(bound_new) :])
                        return
        if self.has_pystra and node.attr in DATA["methods"] and not _symbol(path):
            replacements = DATA["methods"][node.attr]
            message = f"Review {node.attr}: instance methods, getters/results and evaluation shapes are not rewritten."
            if replacements:
                message += " See " + ", ".join(replacements[:2]) + "."
            self.warn(node, message)
        self.generic_visit(node)

    def visit_Call(self, node):
        path = self.resolve(node.func)
        record = _symbol(path) if path else None
        if record and not record["manual"]:
            target = record["target"].rsplit(".", 1)[1]
            if target in {
                "CrudeMonteCarlo",
                "ImportanceSampling",
                "DistributionAnalysis",
                "LineSampling",
                "SubsetSimulation",
            }:
                self.simulation = True
            if len(node.args) > record.get("positional_limit", 1000):
                self.warn(
                    node,
                    "Positional configuration may include input_type or moved options; use explicit 2.0 keywords.",
                )
            existing = {keyword.arg for keyword in node.keywords}
            for keyword in node.keywords:
                if keyword.arg == "input_type":
                    self.warn(
                        keyword,
                        "input_type requires manual conversion to explicit native parameters.",
                    )
                elif keyword.arg in record["keywords"]:
                    new = record["keywords"][keyword.arg]
                    if new in existing:
                        self.warn(
                            keyword,
                            f"Both {keyword.arg} and {new} are supplied; resolve the duplicate manually.",
                        )
                    else:
                        start, _ = self.span(keyword)
                        self.edit(start, start + len(keyword.arg), new)
                elif keyword.arg is None:
                    self.warn(
                        keyword, "Expanded constructor keywords require manual review."
                    )
        if path == "numpy.random.seed":
            self.seeds.append(node)
        self.generic_visit(node)


def _convert(source, context=""):
    prefix = context + "\n" if context else ""
    combined = prefix + source
    try:
        tree = ast.parse(combined)
    except SyntaxError as error:
        return Conversion(
            source,
            (
                Diagnostic(
                    max(1, (error.lineno or 1) - prefix.count("\n")),
                    error.offset or 1,
                    "Syntax could not be resolved (possibly IPython syntax); review this code manually.",
                ),
            ),
        )
    editor = _Editor(combined, tree)
    editor.visit(tree)
    if editor.simulation:
        for node in editor.seeds:
            editor.warn(
                node,
                "Global NumPy seeding does not seed 2.0 simulations; pass rng= explicitly.",
            )
    converted = combined
    previous_start = len(combined) + 1
    for (start, end), replacement in sorted(editor.edits.items(), reverse=True):
        if start < len(prefix):
            continue
        if end > previous_start:
            raise RuntimeError("Overlapping migration edits")
        converted = converted[:start] + replacement + converted[end:]
        previous_start = start
    diagnostics = tuple(
        sorted(
            {
                Diagnostic(d.line - prefix.count("\n"), d.column, d.message)
                for d in editor.diagnostics
                if d.line > prefix.count("\n")
            },
            key=lambda d: (d.line, d.column, d.message),
        )
    )
    return Conversion(converted[len(prefix) :], diagnostics)


def convert_source(source: str) -> Conversion:
    """Convert a Python source string, preserving formatting and local names.

    Only resolved PySTRA imports, module attributes and constructor keywords
    are edited. Shadowed or reassigned bindings are left alone. Diagnostics
    flag manual changes; conversion does not promise that a script is runnable.
    Syntax that cannot be parsed is returned unchanged with a diagnostic.
    A second pass is checked to make no further changes.
    """
    result = _convert(source)
    if _convert(result.source).source != result.source:
        raise RuntimeError("Migration edits are not idempotent")
    return result


def _json_spans(source):
    """Yield JSON value spans and paths without reserializing notebook metadata."""
    decoder = json.JSONDecoder()

    def walk(position, path):
        while source[position].isspace():
            position += 1
        start = position
        if source[position] == "{":
            position += 1
            while True:
                while source[position].isspace():
                    position += 1
                if source[position] == "}":
                    return position + 1
                key, position = decoder.raw_decode(source, position)
                while source[position].isspace() or source[position] == ":":
                    position += 1
                position = yield from walk(position, (*path, key))
                while source[position].isspace():
                    position += 1
                if source[position] == ",":
                    position += 1
        elif source[position] == "[":
            position += 1
            index = 0
            while True:
                while source[position].isspace():
                    position += 1
                if source[position] == "]":
                    position += 1
                    break
                position = yield from walk(position, (*path, index))
                index += 1
                while source[position].isspace():
                    position += 1
                if source[position] == ",":
                    position += 1
        else:
            _, position = decoder.raw_decode(source, position)
        yield path, start, position
        return position

    yield from walk(0, ())


def _notebook(source):
    notebook = json.loads(source)
    edits = {}
    diagnostics = []
    context = ""
    cells = notebook["cells"]
    origins = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code = cell.get("source", "")
        text = "".join(code) if isinstance(code, list) else code
        result = _convert(text, context)
        diagnostics.extend(
            Diagnostic(d.line, d.column, d.message, index + 1)
            for d in result.diagnostics
        )
        if text != result.source:
            edits[("cells", index, "source")] = (
                result.source.splitlines(keepends=True)
                if isinstance(code, list)
                else result.source
            )
        try:
            ast.parse(text)
        except SyntaxError:
            continue
        start_line = context.count("\n") + 2
        origins.append((start_line, start_line + text.count("\n"), index + 1))
        context += "\n" + text
    # A seed in an earlier cell can precede a simulation in a later cell.
    for diagnostic in _convert(context).diagnostics:
        if "Global NumPy seeding" not in diagnostic.message:
            continue
        for first, last, cell_number in origins:
            if first <= diagnostic.line <= last:
                diagnostics.append(
                    Diagnostic(
                        diagnostic.line - first + 1,
                        diagnostic.column,
                        diagnostic.message,
                        cell_number,
                    )
                )
                break
    diagnostics = sorted(
        set(diagnostics), key=lambda d: (d.cell, d.line, d.column, d.message)
    )
    spans = [
        (start, end, json.dumps(edits[path], ensure_ascii=False))
        for path, start, end in _json_spans(source)
        if path in edits
    ]
    for start, end, replacement in reversed(spans):
        source = source[:start] + replacement + source[end:]
    return Conversion(source, tuple(diagnostics))


def convert_notebook(source: str) -> Conversion:
    """Convert notebook code cells in document order, preserving other JSON.

    Markdown, outputs and metadata are untouched. Only changed code-cell
    source fields are serialized. Imports carry across earlier valid Python
    cells; IPython syntax is left unchanged and flagged. Cell execution order
    at runtime is not inferred. Diagnostics use one-based cell numbers.
    """
    result = _notebook(source)
    if _notebook(result.source).source != result.source:
        raise RuntimeError("Notebook migration edits are not idempotent")
    return result
