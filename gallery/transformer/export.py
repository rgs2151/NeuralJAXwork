"""Export cyclic rolling-context attention from the cached character transformer."""
import hashlib
import json
from pathlib import Path
import re
import jax.numpy as jnp
import numpy as np
from neuraljaxlib import Rngs, checkpoint, models, nn, jit

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/gallery/transformer'
model = models.TinyLanguageModel(128,128,width=128,depth=3,heads=2,rngs=Rngs(params=0))
checkpoint.restore(RUN/'best.npz',model=model)
model.eval()
words = re.findall(r'\S+', (RUN/'generation.txt').read_text().split('\n',1)[1])
period = ' '.join(words)+' '
text = period*3
matches = list(re.finditer(r'\S+',text))

@jit
def inspect(candidate,tokens):
    x = candidate.token(tokens)+candidate.position(jnp.arange(128))[None]
    mask = nn.attention.causal_mask(128)
    for block in candidate.blocks:
        z = block.norm1(x)
        output, weights = block.attention(z,z,z,mask=mask,need_weights=True)
        x = x+output
        x = x+block.ff(block.norm2(x))
    return candidate.head(candidate.norm(x)), weights

steps=[]
for i in range(len(words)):
    current=len(words)+i
    end=matches[current].end()
    start=end-128
    tokens=jnp.asarray([[ord(c) for c in text[start:end]]],dtype=jnp.int32)
    logits, weights=inspect(model,tokens)
    if i in (0,len(words)-1):
        np.testing.assert_allclose(np.asarray(logits),np.asarray(model(tokens)),atol=2e-5,rtol=1e-5)
    weights=np.asarray(weights)[0]
    np.testing.assert_allclose(weights.sum(axis=-1),1,atol=1e-6)
    assert np.all(weights[:,np.triu_indices(128,1)[0],np.triu_indices(128,1)[1]]==0)
    rows=[]
    for lag in range(32):
        match=matches[current-lag]
        a=max(0,match.start()-start)
        b=max(0,min(128,matches[current-lag+1].start()-start))
        rows.append(weights[:,-1,a:b].sum(axis=1).tolist())
    heads=np.asarray(rows).T
    np.testing.assert_allclose(heads.sum(axis=1),1,atol=1e-6)
    steps.append(dict(heads=heads.tolist(),mean=heads.mean(axis=0).tolist()))
result=dict(schema=2,layer=3,heads=[1,2],words=words,steps=steps,
            checkpointSha256=hashlib.sha256((RUN/'best.npz').read_bytes()).hexdigest(),
            note='Final-layer heads averaged. Lag zero is current-word mass, omitted from arcs. Character tokens grouped into words. Replayed cached text with whitespace normalized; repeated text supplies the 128-character rolling context across the seamless loop boundary.')
Path(__file__).with_name('samples.js').write_text('window.TRANSFORMER_SAMPLES = '+json.dumps(result,separators=(',',':'))+';\n')
print(f'Exported {len(words)} rolling attention rows; checked normalization, causality and logits.')
