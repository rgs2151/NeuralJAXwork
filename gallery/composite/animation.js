/* Compose frozen renderers into one SVG; individual asset sources stay untouched. */
(function(){
  const NS='http://www.w3.org/2000/svg';
  window.mountGallery=function(root){
    const specs=[
      ['cnn',mountCNN,CNN_SAMPLES,216,216,'#000'],
      ['rnn',mountRNN,RNN_SAMPLES,201*254/214,201,'#fff'],
      ['autoencoder',mountAutoencoder,AE_SAMPLES,216*240/218,216,'#fff'],
      ['gan',mountGAN,GAN_SAMPLES,216,216,'#000'],
      ['transformer',mountTransformer,TRANSFORMER_SAMPLES,216,216,'#fff']
    ];
    const gap=6,height=240,width=specs.reduce((sum,s)=>sum+s[3],0)+gap*(specs.length-1);
    const svg=document.createElementNS(NS,'svg');
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('role','img');
    svg.setAttribute('aria-label','NeuralJAXlib gallery: CNN, RNN, Autoencoder, GAN, Transformer.');
    svg.style.cssText='display:block;width:100%;height:auto;background:#fff;color:#171717;font-family:Arial,sans-serif';root.append(svg);
    function el(tag,attrs){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);svg.append(n);return n;}
    const mounts=[],controllers=[];let x=0;
    specs.forEach(([name,mount,data,w,h,bg])=>{
      const host=document.createElement('div');host.style.display='none';root.append(host);mounts.push(host);
      const controller=mount(host,data);controller.pause();controllers.push(controller);
      const title=host.firstElementChild.textContent;
      el('text',{x:x+w/2,y:18,'text-anchor':'middle','font-family':'Arial,sans-serif','font-size':18,'font-weight':700,fill:'#171717'}).textContent=title;
      el('rect',{x,y:24,width:w,height:h,fill:bg});
      const plot=host.querySelector('svg');plot.setAttribute('x',x);plot.setAttribute('y',24);
      plot.setAttribute('width',w);plot.setAttribute('height',h);
      plot.style.width=w+'px';plot.style.height=h+'px';svg.append(plot);
      if(name==='rnn'){
        // Nested SVG bounding boxes reflect artwork, not the allocated viewport.
        // Override the standalone resize attribute with a fixed composite size.
        plot.querySelectorAll('text').forEach(node=>node.style.fontSize=(18*254/w)+'px');
        const legend=host.lastElementChild;
        const labels=[...legend.children].map(item=>item.textContent);
        const widths=labels.map(label=>label==='Forecast'?92:50),total=widths[0]+14+widths[1];let left=x+(w-total)/2;
        labels.forEach((label,i)=>{el('circle',{cx:left+4,cy:233,r:4,fill:i===0?'#8b1e1e':'#000'});el('text',{x:left+14,y:239,'font-family':'Arial,sans-serif','font-size':18,fill:'#171717'}).textContent=label;left+=widths[i]+14;});
      }
      x+=w+gap;
    });
    let elapsed=0,last=null,paused=false,disposed=false,raf=0;
    const motion=matchMedia('(prefers-reduced-motion: reduce)');
    function draw(ms){controllers.forEach(c=>c.seek(ms));}
    function tick(now){if(disposed)return;if(last!==null&&!paused&&!document.hidden&&!motion.matches){elapsed+=Math.min(now-last,100);draw(elapsed);}last=now;raf=requestAnimationFrame(tick);}
    function preference(){draw(motion.matches?14000:elapsed);}
    motion.addEventListener('change',preference);preference();raf=requestAnimationFrame(tick);
    return {width,height,seek(ms){elapsed=ms;paused=true;draw(ms);},play(){paused=false;},pause(){paused=true;},destroy(){disposed=true;cancelAnimationFrame(raf);motion.removeEventListener('change',preference);controllers.forEach(c=>c.destroy());mounts.forEach(h=>h.remove());svg.remove();}};
  };
})();
