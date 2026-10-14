/* Continuous word stream with mean causal attention below the text. */
(function(){
  const NS='http://www.w3.org/2000/svg';
  window.mountTransformer=function(root,data){
    const heading=document.createElement('div');heading.textContent='nj.Module.Transformer';
    heading.style.cssText='font:700 18px/1.05 Arial,sans-serif;text-align:center;margin:0;color:inherit';
    const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 240 240');svg.setAttribute('role','img');
    svg.style.cssText='display:block;width:100%;height:auto;overflow:hidden;background:#fff;font-family:Arial,sans-serif';root.append(heading,svg);
    function el(tag,attrs,parent=svg){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);parent.append(n);return n;}
    const arcs=el('g',{'aria-label':'Mean attention to earlier words'}),text=el('g',{});
    const count=32,gap=8,right=232;
    const nodes=Array.from({length:count},()=>el('text',{y:34,'text-anchor':'end','font-size':18,fill:'#222'},text));
    const paths=Array.from({length:count-1},()=>el('path',{fill:'none',stroke:'#000','stroke-linecap':'round'},arcs));
    const context=document.createElement('canvas').getContext('2d');context.font='700 18px Arial';
    const widths=data.words.map(w=>context.measureText(w).width);
    const stepDuration=850,duration=stepDuration*data.words.length;
    const wrap=i=>(i%data.words.length+data.words.length)%data.words.length;
    function draw(ms,final=false){
      const time=Math.max(0,ms)%duration,current=Math.floor(time/stepDuration);
      const t=final?1:Math.min(1,(time%stepDuration)/450),ease=t*t*(3-2*t);
      let edge=right+(1-ease)*(widths[current]+gap);
      const centers=[];
      nodes.forEach((node,lag)=>{
        const index=wrap(current-lag);
        node.textContent=data.words[index];node.setAttribute('x',edge);
        node.style.visibility=lag===0&&ease===0?'hidden':'visible';
        centers.push(edge-widths[index]/2);
        // The outgoing current word loses its highlight as the incoming one arrives.
        const red=lag===0?1:lag===1?1-ease:0;
        node.setAttribute('fill',red===1?'#8b1e1e':`rgb(${Math.round(34+105*red)},${Math.round(34-4*red)},${Math.round(34-4*red)})`);
        node.setAttribute('font-weight',lag===0||lag===1&&ease===0?'700':'400');
        node.dataset.lag=String(lag);
        edge-=widths[index]+gap;
      });
      const from=centers[1]+ease*(centers[0]-centers[1]);
      paths.forEach((path,i)=>{
        const lag=i+1,to=centers[lag];
        const oldWeight=lag===1?0:data.steps[wrap(current-1)].mean[lag-1];
        const weight=oldWeight+(data.steps[current].mean[lag]-oldWeight)*ease;
        const radius=(from-to)/2;
        const visible=to>-16&&to<240&&radius>0.1&&weight>0;
        path.style.display=visible?'':'none';
        if(!visible)return;
        path.setAttribute('d',`M ${from} 48 A ${radius} ${radius} 0 0 1 ${to} 48`);
        path.setAttribute('stroke-width',0.8+11.2*weight);
        path.setAttribute('opacity',Math.max(0,Math.min(1,(to+16)/28)));
        path.dataset.weight=String(weight);path.dataset.lag=String(lag);
      });
      svg.setAttribute('aria-label',`Transformer layer ${data.layer}, mean of heads ${data.heads.join(' and ')}. Current word: ${data.words[current]}. Black arcs below show attention to preceding word spans.`);
      root.dataset.word=String(current);root.dataset.progress=ease.toFixed(4);
    }
    let elapsed=0,last=null,raf=0,paused=false,visible=true,disposed=false;
    const motion=matchMedia('(prefers-reduced-motion: reduce)');function preference(){draw(elapsed,motion.matches);}
    const observer=new IntersectionObserver(es=>{visible=es[0].isIntersecting;});observer.observe(root);
    function tick(now){if(disposed)return;if(last!==null&&!paused&&visible&&!document.hidden&&!motion.matches){elapsed+=Math.min(now-last,100);draw(elapsed);}last=now;raf=requestAnimationFrame(tick);}
    motion.addEventListener('change',preference);preference();raf=requestAnimationFrame(tick);
    return {duration,stepDuration,seek(ms){elapsed=ms;paused=true;draw(ms);},pause(){paused=true;},play(){paused=false;},destroy(){disposed=true;cancelAnimationFrame(raf);observer.disconnect();motion.removeEventListener('change',preference);heading.remove();svg.remove();}};
  };
})();
