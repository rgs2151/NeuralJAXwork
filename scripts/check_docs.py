"""Check the built book contains all tutorial chapters, generated images and sources."""

import sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(__file__).resolve().parents[1]
chapters = sorted((source / "notebooks").glob("*.ipynb"))
index = (root / "docs/tutorials.html").read_text()
for chapter in chapters:
    page = root / "notebooks" / f"{chapter.stem}.html"
    html = page.read_text()
    assert f"{chapter.stem}.html" in index, f"Missing navigation: {chapter.name}"
    assert "cell_output" in html, f"Missing executed output: {chapter.name}"
    assert "<img" in html and "_images/" in html, f"Missing generated figure: {chapter.name}"
    assert list((root / "_downloads").rglob(chapter.name)), (
        f"Missing source download: {chapter.name}"
    )
assert (root / ".nojekyll").exists(), "Pages must preserve Sphinx underscore directories"
print(f"Book verified: {len(chapters)} linked, executed tutorials with figures and downloads.")
