/* Compact in-place pooling illustration; the final prediction is cached. */
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const SIDE = 280;
  const RED = '#8b1e1e';
  const clamp = value => Math.max(0, Math.min(1, value));
  const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };

  function pool(map) {
    const size = map.size / 2;
    if (!Number.isInteger(size)) throw new Error('Pooling requires an even grid size');
    const values = [];
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const i = row * 2 * map.size + col * 2;
        values.push(Math.max(map.values[i], map.values[i + 1],
          map.values[i + map.size], map.values[i + map.size + 1]));
      }
    }
    return { size, values };
  }

  window.mountCNN = function (root, data) {
    const heading = document.createElement('div');
    heading.textContent = 'nj.Module.CNN';
    heading.style.cssText = 'font:700 18px/1.2 Arial,sans-serif;text-align:center;margin:0 0 6px;color:inherit';
    root.append(heading);
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${SIDE} ${SIDE}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'A brain MRI slice is pooled illustratively in place, then replaced by its cached tumor-type prediction.');
    svg.style.cssText = 'display:block;width:100%;height:auto;background:#000';
    root.append(svg);
    function el(tag, attrs, parent = svg) {
      const node = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
      parent.append(node);
      return node;
    }
    const pixels = el('g', {});
    const cursor = el('rect', {
      fill: RED, 'fill-opacity': 0.18, stroke: RED, 'stroke-width': 4.5,
      'vector-effect': 'non-scaling-stroke', visibility: 'hidden',
    });
    const result = el('text', {
      x: SIDE / 2, y: SIDE / 2, 'text-anchor': 'middle',
      'dominant-baseline': 'central', 'font-family': 'Arial,sans-serif',
      'font-weight': 700, 'font-size': 32, fill: '#fff', visibility: 'hidden',
    });
    const samples = data.samples.map(sample => {
      const input = sample.maps[0];
      const maps = [input];
      for (let i = 0; i < 4; i++) maps.push(pool(maps.at(-1)));
      return { prediction: sample.prediction, maps };
    });
    // Give finer grids more time without rushing the coarse passes.
    let nextStart = 700;
    const passes = [5600, 3200, 1800, 1200].map(scanDuration => {
      const pass = { start: nextStart, end: nextStart + scanDuration };
      nextStart = pass.end + 350;
      return pass;
    });
    const revealStart = nextStart;
    const duration = revealStart + 600;
    let activeKey = '', cells = [], blocks = [];

    function build(map, pooled) {
      pixels.replaceChildren();
      const step = SIDE / map.size;
      cells = map.values.map((value, i) => el('rect', {
        x: (i % map.size) * step, y: Math.floor(i / map.size) * step,
        width: step, height: step, fill: `rgb(${value},${value},${value})`,
        stroke: '#000', 'stroke-width': 0.45,
      }, pixels));
      // Opaque larger blocks replace each four-cell patch in the same coordinates.
      blocks = pooled.values.map((value, i) => el('rect', {
        x: (i % pooled.size) * step * 2, y: Math.floor(i / pooled.size) * step * 2,
        width: step * 2, height: step * 2, fill: `rgb(${value},${value},${value})`,
        stroke: '#000', 'stroke-width': 0.6, visibility: 'hidden',
      }, pixels));
    }

    function render(milliseconds) {
      const total = Math.max(0, milliseconds);
      const time = total % duration;
      const index = Math.floor(total / duration) % samples.length;
      const sample = samples[index];
      const phase = time >= revealStart ? 4 : Math.max(0, passes.findLastIndex(pass => time >= pass.start));
      root.dataset.stage = String(phase);
      root.dataset.digit = String(sample.prediction);
      result.textContent = sample.prediction;
      result.setAttribute('visibility', phase === 4 ? 'visible' : 'hidden');
      pixels.style.display = phase === 4 ? 'none' : '';
      cursor.setAttribute('visibility', 'hidden');
      const layer = Math.min(phase, 3);
      const key = `${index}:${layer}`;
      const map = sample.maps[layer], pooled = sample.maps[layer + 1];
      if (key !== activeKey) { build(map, pooled); activeKey = key; }
      const pass = passes[layer];
      const position = clamp((time - pass.start) / (pass.end - pass.start)) * blocks.length;
      const done = Math.floor(position), fraction = position - done;
      // Alternating row direction keeps the window moving continuously at row ends.
      const order = i => {
        const row = Math.floor(i / pooled.size), col = i % pooled.size;
        return row * pooled.size + (row % 2 ? pooled.size - 1 - col : col);
      };
      blocks.forEach((block, i) => {
        const rank = order(i);
        const alpha = rank < done ? 1 : rank === done ? ease(fraction) : 0;
        block.setAttribute('visibility', alpha > 0 ? 'visible' : 'hidden');
        block.setAttribute('opacity', alpha);
      });
      cells.forEach((cell, i) => {
        const row = Math.floor(i / map.size), col = i % map.size;
        const patch = Math.floor(row / 2) * pooled.size + Math.floor(col / 2);
        cell.setAttribute('visibility', order(patch) < done ? 'hidden' : 'visible');
      });
      if (time >= pass.start && time < pass.end) {
        const width = SIDE / pooled.size;
        const here = order(done), next = order(Math.min(done + 1, blocks.length - 1));
        const x = (here % pooled.size) + ((next % pooled.size) - (here % pooled.size)) * fraction;
        const y = Math.floor(here / pooled.size) + (Math.floor(next / pooled.size) - Math.floor(here / pooled.size)) * fraction;
        cursor.setAttribute('x', x * width + 2.25);
        cursor.setAttribute('y', y * width + 2.25);
        cursor.setAttribute('width', width - 4.5);
        cursor.setAttribute('height', width - 4.5);
        cursor.setAttribute('visibility', 'visible');
      }
    }

    let elapsed = 0, last = null, raf = 0, disposed = false, visible = true, paused = false;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    function tick(now) {
      if (disposed) return;
      if (last !== null && visible && !document.hidden && !paused && !motion.matches) {
        elapsed += Math.min(now - last, 100);
        render(elapsed);
      }
      last = now;
      raf = requestAnimationFrame(tick);
    }
    function preference() { render(motion.matches ? revealStart + 300 : elapsed); }
    motion.addEventListener('change', preference);
    const observer = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; });
    observer.observe(root);
    preference(); raf = requestAnimationFrame(tick);
    return {
      duration,
      seek(ms) { paused = true; elapsed = ms; render(ms); },
      play() { paused = false; },
      pause() { paused = true; },
      destroy() {
        disposed = true; cancelAnimationFrame(raf); observer.disconnect();
        motion.removeEventListener('change', preference); heading.remove(); svg.remove();
      },
    };
  };
})();
