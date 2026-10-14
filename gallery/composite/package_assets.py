"""Package the five frozen animations and composite as standalone HTML files."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'frozen.json').read_text())
for file,digest in manifest['sha256'].items():
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest, f'Frozen asset changed: {file}'
output=ROOT/'assets'
output.mkdir(exist_ok=True)
for name in [*manifest['assets'],'composite']:
    target=output/f'{name}.html'
    subprocess.run([sys.executable,str(ROOT/name/'build_preview.py'),str(target)],check=True)
    fragment=target.read_text()
    target.write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NeuralJAXlib '+name+'</title><style>body{margin:0;background:white;color:#171717;font-family:Arial,sans-serif}</style></head><body>\n'+fragment+'\n</body></html>\n')
print('Frozen sources verified; standalone assets packaged.')
