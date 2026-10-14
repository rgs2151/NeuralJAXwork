"""Blend the composite excerpt's last eight frames into its opening frame."""

import subprocess
from pathlib import Path

from PIL import Image, ImageSequence

path = Path(__file__).with_name("gallery.gif")
with Image.open(path) as gif:
    palette = gif.copy()
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif)]
    gif.seek(0)
    durations = [frame.info.get("duration", 70) for frame in ImageSequence.Iterator(gif)]
for i in range(8):
    frames[-8 + i] = Image.blend(frames[-8 + i], frames[0], (i + 1) / 8)
frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
for frame in frames:
    frame.info.clear()
output = path.with_suffix(".tmp.gif")
frames[0].save(
    output,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=False,
    disposal=2,
)

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(output),
        "-filter_complex",
        "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
        "-loop",
        "0",
        str(path),
    ],
    check=True,
)
output.unlink()
