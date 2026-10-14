/* Fixed-latent Fashion-MNIST training progression, rendered as one SVG pixel grid. */
(function () {
  const NS='http://www.w3.org/2000/svg';
  window.mountGAN=function(root,data){
    const heading=document.createElement('div');
    heading.textContent='nj.Module.GAN';
    heading.style.cssText='font:700 18px/1.2 Arial,sans-serif;text-align:center;margin:0 0 2px;color:inherit';
    const svg=document.createElementNS(NS,'svg');
    const batchSize=4,gap=1,extent=data.size*2+gap;
    svg.setAttribute('viewBox',`0 0 ${extent} ${extent}`);
    svg.setAttribute('role','img');
    svg.setAttribute('aria-label','Fashion-MNIST GAN: batches of four fixed-noise samples developing from epoch three to thirty in a two-by-two grid.');
    svg.style.cssText='display:block;width:100%;height:auto;background:#000;shape-rendering:crispEdges';
    root.append(heading,svg);
    const area=data.size*data.size;
    const frames=data.samples.map(sample=>Uint8Array.from(atob(sample.pixels),c=>c.charCodeAt(0)));
    if(frames.some(frame=>frame.length!==data.frames*area))throw new Error('Invalid GAN frame data');
    if(!frames.length||frames.length%batchSize)throw new Error('GAN samples must form complete batches of four');
    const cells=Array.from({length:area*batchSize},(_,i)=>{
      const tile=Math.floor(i/area),pixel=i%area;
      const cell=document.createElementNS(NS,'rect');
      cell.setAttribute('x',(tile%2)*(data.size+gap)+pixel%data.size);
      cell.setAttribute('y',Math.floor(tile/2)*(data.size+gap)+Math.floor(pixel/data.size));
      cell.setAttribute('width',1);cell.setAttribute('height',1);svg.append(cell);return cell;
    });
    const lastPixels=new Int16Array(area*batchSize).fill(-1);
    const startFrame=Math.min(3,data.frames-1);
    const morphDuration=1700*(data.frames-1-startFrame)/(data.frames-1);
    const sampleDuration=morphDuration,duration=sampleDuration*frames.length/batchSize;
    function draw(ms){
      const time=Math.max(0,ms)%duration;
      const index=Math.floor(time/sampleDuration),local=time%sampleDuration;
      const progress=Math.max(0,Math.min(1,local/morphDuration));
      const position=startFrame+progress*(data.frames-1-startFrame),first=Math.floor(position),second=Math.min(first+1,data.frames-1);
      const fraction=position-first;
      for(let i=0;i<cells.length;i++){
        const frame=frames[index*batchSize+Math.floor(i/area)],pixel=i%area;
        const value=Math.round(frame[first*area+pixel]*(1-fraction)+frame[second*area+pixel]*fraction);
        if(value!==lastPixels[i]){cells[i].setAttribute('fill',`rgb(${value},${value},${value})`);lastPixels[i]=value;}
      }
      root.dataset.batch=String(index);
      root.dataset.samples=data.samples.slice(index*batchSize,(index+1)*batchSize).map(sample=>sample.index).join(',');
      root.dataset.epoch=position.toFixed(2);
    }
    let elapsed=0,last=null,raf=0,paused=false,visible=true,disposed=false,lastDraw=-Infinity;
    const motion=matchMedia('(prefers-reduced-motion: reduce)');
    function preference(){draw(motion.matches?morphDuration-0.001:elapsed);}
    const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;});observer.observe(root);
    function tick(now){
      if(disposed)return;
      if(last!==null&&!paused&&visible&&!document.hidden&&!motion.matches){
        elapsed+=Math.min(now-last,100);
        if(now-lastDraw>=1000/30){draw(elapsed);lastDraw=now;}
      }
      last=now;raf=requestAnimationFrame(tick);
    }
    motion.addEventListener('change',preference);preference();raf=requestAnimationFrame(tick);
    return {duration,sampleDuration,seek(ms){elapsed=ms;paused=true;draw(ms);},pause(){paused=true;},play(){paused=false;},destroy(){disposed=true;cancelAnimationFrame(raf);observer.disconnect();motion.removeEventListener('change',preference);heading.remove();svg.remove();}};
  };
})();
