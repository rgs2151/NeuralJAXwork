/* Compose intact standalone panels at one design scale, without nested SVG rescaling. */
(function () {
  window.mountGallery = function (root) {
    const specs = [
      ['cnn', mountCNN, CNN_SAMPLES],
      ['rnn', mountRNN, RNN_SAMPLES],
      ['autoencoder', mountAutoencoder, AE_SAMPLES],
      ['gan', mountGAN, GAN_SAMPLES],
      ['transformer', mountTransformer, TRANSFORMER_SAMPLES],
    ];
    const panelWidth = 240, gap = 12, width = panelWidth * specs.length + gap * (specs.length - 1), height = 270;
    const viewport = document.createElement('div');
    viewport.setAttribute('role', 'group');
    viewport.setAttribute('aria-label', 'NeuralJAXlib gallery: CNN, RNN, Autoencoder, GAN, Transformer.');
    viewport.style.cssText = 'position:relative;width:100%;color:#171717;font-family:Arial,sans-serif';
    const row = document.createElement('div');
    row.dataset.galleryRow = '';
    row.style.cssText = `position:absolute;left:0;top:0;display:grid;grid-template-columns:repeat(5,${panelWidth}px);gap:${gap}px;width:${width}px;height:${height}px;transform-origin:top left`;
    viewport.append(row); root.append(viewport);
    const controllers = specs.map(([name, mount, data]) => {
      const panel = document.createElement('div');
      panel.dataset.galleryPanel = name;
      panel.style.cssText = 'display:grid;grid-template-rows:24px 240px;row-gap:6px;min-width:0;align-items:center';
      row.append(panel);
      const controller = mount(panel, data);
      controller.pause();
      const heading = panel.firstElementChild;
      heading.style.cssText = 'grid-row:1;margin:0;font:700 18px/24px Arial,sans-serif;text-align:center;color:inherit;white-space:nowrap';
      const plot = panel.querySelector('svg');
      plot.style.gridRow = '2';
      // Match PCA labels to the 18px RNN legend and transformer words.
      if (name === 'autoencoder') plot.querySelectorAll('text').forEach(node => { node.style.fontSize = '18px'; });
      if (name === 'rnn') {
        panel.style.gridTemplateRows = '24px auto 24px';
        panel.style.alignContent = 'start';
        const legend = panel.lastElementChild;
        legend.style.gridRow = '3'; legend.style.margin = '0'; legend.style.lineHeight = '24px';
      }
      return controller;
    });
    const resize = new ResizeObserver(() => {
      const scale = (root.clientWidth || width) / width;
      row.style.transform = `scale(${scale})`;
      viewport.style.height = `${height * scale}px`;
    });
    resize.observe(root);
    let elapsed = 0, last = null, paused = false, disposed = false, visible = true, raf = 0;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const observer = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; });
    observer.observe(root);
    function draw(ms) { controllers.forEach(controller => controller.seek(ms)); }
    function tick(now) {
      if (disposed) return;
      if (last !== null && !paused && visible && !document.hidden && !motion.matches) {
        elapsed += Math.min(now - last, 100); draw(elapsed);
      }
      last = now; raf = requestAnimationFrame(tick);
    }
    function preference() { draw(motion.matches ? 14000 : elapsed); }
    motion.addEventListener('change', preference); preference(); raf = requestAnimationFrame(tick);
    return {
      width, height,
      seek(ms) { elapsed = ms; paused = true; draw(ms); },
      play() { paused = false; },
      pause() { paused = true; },
      destroy() {
        disposed = true; cancelAnimationFrame(raf); resize.disconnect(); observer.disconnect();
        motion.removeEventListener('change', preference);
        controllers.forEach(controller => controller.destroy()); viewport.remove();
      },
    };
  };
})();
