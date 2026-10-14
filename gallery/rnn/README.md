# RNN weather clock

The resize callback uses the SVG's untransformed `clientWidth`. Labels therefore
scale with the artwork when a host applies CSS scaling or rotation; they do not
grow to compensate for a small preview. This fixes the oversized "now" label.

Open `index.html` for the SVG/JavaScript animation. The tightly cropped clock is 300×253 pixels at its default width and scales to 240×202 pixels; the heading is above the square and the two-item legend is below it. Labels use the heading’s 18-pixel font size at normal weight. All strokes use a shared 2-pixel width with rounded joins and butt caps on the fading traces to avoid dark overlaps and remain the same thickness when resized. The CNN animation is unchanged.

The clock is rotated 90 degrees clockwise, placing the output line at the right. A horizontal now label sits above and to the right of that line. Increasing clockwise angle means increasing age: one full turn spans 24 days. Day and temperature-unit labels are omitted. Temperature increases radially outwards on a fixed Celsius scale, from radius 66 at the neuron circle’s circumference to radius 102.5 at the previous band’s midpoint. There is no outer grid. Both traces begin filled; the newest comparison stays at three o’clock as history moves clockwise. One 24-day revolution takes 18 seconds, then the excerpt resets. The historical seam is open rather than connecting old and new values. Both traces fade linearly with age, from fully opaque at now to transparent at the oldest end, using 96 short SVG sections.

The legend reads Forecast and Real: dark red denotes forecasts, black denotes observations. The data are a fixed, chronological excerpt from January 1–February 18, 2024 (1,176 consecutive hourly samples), exported from the cached NYC weather test predictions. These are archived daily 24-hour forecasts, compared retrospectively with realized temperatures at each target hour—not new one-hour forecasts issued at every animation step. Smooth interpolation between hourly observations is for playback only.

The center follows the supplied recurrent-network reference, using a white interior with a black border, curved connections, and white-to-dark-red nodes. Neuron colors are illustrative, not measured firing rates or hidden-state traces. A plain red output connection indicates the forecast comparison at now. `nj.Module.RNN` is a presentation label, not an import path.

No training or network access is needed for playback. `samples.js` records timestamps, forecast issue times, and the source artifact hash. Regenerate with `.venv/bin/python gallery/rnn/export.py`; bundle for chat with `python3 gallery/rnn/build_preview.py /absolute/output/rnn-clock.html`.

`mountRNN(element, RNN_SAMPLES)` returns `seek(milliseconds)`, `pause()`, `play()`, and `destroy()`. Reduced-motion users get a static clock; playback pauses offscreen and in background tabs.
