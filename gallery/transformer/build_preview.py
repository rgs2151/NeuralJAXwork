"""Bundle the standalone animation as a self-contained chat preview."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = Path(__file__).parent
parts = [
    '<div id="transformer-gallery-preview" style="width:100%;max-width:240px;margin:auto"></div>\n',
    "<script>\n",
    (source / "samples.js").read_text(),
    (source / "animation.js").read_text(),
    "window.transformer = mountTransformer(document.getElementById('transformer-gallery-preview'), TRANSFORMER_SAMPLES);\n",
    "</script>\n",
]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("".join(parts))
print(args.output)
