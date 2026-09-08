"""Check generated local links, tutorial downloads and figure alternatives."""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from execute_notebooks import indexed_notebooks


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.anchors = set()
        self.links = []
        self.images = []
        self.text = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_data(self, data):
        self.text.append(data)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "id" in attributes:
            self.anchors.add(attributes["id"])
        if tag == "a":
            if "name" in attributes:
                self.anchors.add(attributes["name"])
            if "href" in attributes:
                self.links.append(attributes["href"])
        if tag == "img":
            self.images.append(attributes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path, help="Sphinx HTML build directory")
    args = parser.parse_args()
    root = args.html.resolve()
    source = Path(__file__).resolve().parents[1] / "docs/source"
    pages = {path: Page(path) for path in root.rglob("*.html")}
    issues = []
    links = 0
    for path, page in pages.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc or href.startswith("/"):
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            links += 1
            if not target.is_file():
                issues.append(f"{path.relative_to(root)}: missing file {href}")
            elif url.fragment and target in pages:
                if unquote(url.fragment) not in pages[target].anchors:
                    issues.append(f"{path.relative_to(root)}: missing anchor {href}")
    for name in ("index.html", "install.html"):
        page = pages.get(root / name)
        if page is None:
            issues.append(f"Missing entry page: {name}")
            continue
        content = "".join(page.text)
        if any(token in content for token in ("|release|", ":doc:", ".. _")):
            issues.append(f"{name}: unrendered documentation markup")
    figures = 0
    notebooks = indexed_notebooks(source / "tutorial.rst")
    for notebook in notebooks:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        path = root / notebook.relative_to(source).with_suffix(".html")
        if path not in pages:
            issues.append(f"Missing tutorial page: {path}")
            continue
        page = pages[path]
        for cell in document["cells"]:
            if cell["cell_type"] == "markdown":
                for alias in re.findall(
                    r'<a id="([^"]+)"></a>', "".join(cell["source"])
                ):
                    if alias not in page.anchors:
                        issues.append(
                            f"{notebook.stem}: missing retained anchor {alias}"
                        )
        downloads = {}
        for href in page.links:
            target = (path.parent / unquote(urlsplit(href).path)).resolve()
            if target.suffix in {".ipynb", ".zip"} and target.is_file():
                downloads[target.suffix] = target
        if set(downloads) != {".ipynb", ".zip"}:
            issues.append(f"{notebook.stem}: missing notebook or bundle download")
            continue
        if downloads[".ipynb"].read_bytes() != notebook.read_bytes():
            issues.append(f"{notebook.stem}: stale notebook download")
        with ZipFile(downloads[".zip"]) as bundle:
            if bundle.read(notebook.name) != notebook.read_bytes():
                issues.append(f"{notebook.stem}: stale notebook in bundle")
            for helper in document["metadata"]["pystra"]["support_files"]:
                if bundle.read(helper) != (notebook.parent / helper).read_bytes():
                    issues.append(f"{notebook.stem}: stale helper {helper}")
            if not {"README.txt", "LICENSE"}.issubset(bundle.namelist()):
                issues.append(
                    f"{notebook.stem}: missing bundle instructions or licence"
                )
        expected = [
            alt
            for cell in document["cells"]
            for alt in cell["metadata"].get("pystra", {}).get("figure_alts", [])
        ]
        actual = [
            attributes.get("alt", "")
            for attributes in page.images
            if f"notebooks_{notebook.stem}_" in attributes.get("src", "")
        ]
        figures += len(actual)
        if actual != expected:
            issues.append(f"{notebook.stem}: missing or mismatched figure alternatives")
    for issue in sorted(set(issues)):
        print(issue)
    print(
        f"Checked {len(pages)} pages, {links} local links, "
        f"{len(notebooks)} notebook bundles and {figures} figures: "
        f"{len(set(issues))} issues"
    )
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
