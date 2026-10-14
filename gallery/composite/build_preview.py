"""Bundle the composite with the five canonical renderers for an inline preview."""
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
parts=['<div id="gallery-composite" style="width:100%;margin:auto"></div>\n<script>\n']
for name in ['cnn','rnn','transformer','autoencoder','gan']:
    parts.extend((root/name/file).read_text() for file in ['samples.js','animation.js'])
parts.append(Path(__file__).with_name('animation.js').read_text())
parts.append("window.gallery=mountGallery(document.getElementById('gallery-composite'));\n</script>\n")
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(''.join(parts))
print(args.output)
