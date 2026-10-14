"""Validate saved notebooks and their rendered pages without executing code."""

import ast
import hashlib
import re
import sys
from pathlib import Path

import nbformat

root = Path(sys.argv[1])
source = Path(__file__).resolve().parents[1]
configuration = ast.parse((source / "conf.py").read_text())
modes = [
    node.value.value
    for node in configuration.body
    if isinstance(node, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "nb_execution_mode" for t in node.targets)
]
assert modes == ["off"], "Documentation builds must never execute notebooks"
indices = "".join(
    (root / "docs" / name).read_text() for name in ("learning.html", "api_tutorials.html")
)
chapters = sorted((source / "notebooks").glob("*.ipynb"))
for chapter in chapters:
    notebook = nbformat.read(chapter, as_version=4)
    nbformat.validate(notebook)
    metadata = notebook.metadata.get("neuraljaxlib", {})
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code), (
        f"Unexecuted local cell: {chapter.name}"
    )
    assert not any(o.output_type == "error" for c in code for o in c.outputs), (
        f"Saved error: {chapter.name}"
    )
    digest = hashlib.sha256("\n".join(c.source for c in code).encode()).hexdigest()
    assert digest == metadata.get("source_sha256"), (
        f"Stale saved outputs: {chapter.name}; execute locally again"
    )
    html = (root / "notebooks" / f"{chapter.stem}.html").read_text()
    assert f"{chapter.stem}.html" in indices, f"Missing navigation: {chapter.name}"
    assert "cell_output" in html, f"Missing saved outputs: {chapter.name}"
    if metadata.get("requires_figure"):
        assert "<img" in html and "_images/" in html, f"Missing figure: {chapter.name}"
    assert list((root / "_downloads").rglob(chapter.name)), f"Missing download: {chapter.name}"
assert (root / ".nojekyll").exists()
assert "neuraljaxlib.svg" in (root / "index.html").read_text(), "Project logo is missing"
homepage = (root / "index.html").read_text()
assert re.search(r'<img\b[^>]*src="_images/gallery\.gif"', homepage), (
    "Landing page must display the gallery GIF as an image"
)
assert (root / "_images/gallery.gif").read_bytes() == (
    source / "gallery/composite/gallery.gif"
).read_bytes(), "Published gallery differs from the approved GIF"
for section in ("quickstart", "guide", "learning", "api_tutorials", "api_reference"):
    assert f"docs/{section}.html" in (root / "index.html").read_text()
for internal in ("results", "validation", "publishing", "development"):
    assert not (root / "docs" / f"{internal}.html").exists(), (
        f"Maintainer-only page published: {internal}; build into a clean directory"
    )
assert not (root / "_static" / "results.json").exists(), "Internal run record published"
print(
    f"Validated {len(chapters)} saved notebooks, rendered pages, downloads, and project logo. No code executed."
)
