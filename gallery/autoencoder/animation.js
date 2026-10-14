/* Matched test samples and class means, interpolating input and bottleneck PCA. */
(function(){
  const NS='http://www.w3.org/2000/svg';
  window.mountAutoencoder=function(root,data){
    const heading=document.createElement('div');
    heading.textContent='nj.Module.Autoencoder';
    heading.style.cssText='font:700 18px/1.05 Arial,sans-serif;text-align:center;margin:0;color:inherit';
    const svg=document.createElementNS(NS,'svg');
    svg.setAttribute('viewBox','0 0 240 218');
    svg.setAttribute('role','img');
    svg.style.cssText='display:block;width:100%;height:auto;background:white;font-family:Arial,sans-serif';
    root.append(heading,svg);
    function el(tag,attrs,parent=svg){const node=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);parent.append(node);return node;}
    el('path',{d:'M 12 176 L 12 207 L 45 207 M 8 182 L 12 176 L 16 182 M 39 203 L 45 207 L 39 211',stroke:'#222','stroke-width':1.5,fill:'none'});
    el('text',{x:50,y:212,'font-size':12,fill:'#222'}).textContent='PC1';
    el('text',{x:3,y:168,'font-size':12,fill:'#222'}).textContent='PC2';
    const colors=['#a32939','#277ab3','#20846a','#8a58ad','#ca871c'];
    const plot=el('svg',{x:0,y:0,width:240,height:196,viewBox:'0 0 240 196',overflow:'hidden'});
    const cloud=el('g',{'aria-label':'One hundred held-out samples per class'},plot);
    const dots=Array.from({length:Math.max(...data.batches.map(b=>b.dots.length))},()=>el('circle',{r:1.6,'fill-opacity':0.5},cloud));
    const markers=Array.from({length:5},(_,i)=>{
      const group=el('g',{},plot);
      const image=el('image',{width:24,height:24},group);
      el('circle',{cx:22,cy:22,r:3,fill:colors[i],stroke:'white','stroke-width':1},group);
      return {group,image};
    });
    // Use one fixed, mildly cropped viewport for both endpoints of each batch.
    const views=data.batches.map(batch=>{
      const points=batch.dots.flatMap(s=>[s.start,s.end]);
      const xs=points.map(p=>p[0]).sort((a,b)=>a-b),ys=points.map(p=>p[1]).sort((a,b)=>a-b);
      const low=Math.floor((points.length-1)*0.02),high=Math.ceil((points.length-1)*0.98);
      return {cx:(xs[low]+xs[high])/2,cy:(ys[low]+ys[high])/2,
        scale:200/Math.max(xs[high]-xs[low],ys[high]-ys[low],1e-6)};
    });
    const batchDuration=1800,duration=batchDuration*data.batches.length;
    let shown=-1;
    function draw(ms,final=false){
      const time=Math.max(0,ms)%duration,index=Math.floor(time/batchDuration),local=time%batchDuration;
      const batch=data.batches[index];
      const t=final?1:Math.max(0,Math.min(1,(local-300)/1000));
      const ease=t*t*(3-2*t);
      const {cx,cy,scale}=views[index];
      dots.forEach((dot,i)=>{
        const sample=batch.dots[i];
        dot.style.display=sample?'':'none';
        if(!sample)return;
        if(shown!==index)dot.setAttribute('fill',colors[sample.classIndex]);
        const x=sample.start[0]+ease*(sample.end[0]-sample.start[0]);
        const y=sample.start[1]+ease*(sample.end[1]-sample.start[1]);
        dot.setAttribute('cx',120+(x-cx)*scale);
        dot.setAttribute('cy',98-(y-cy)*scale);
      });
      batch.samples.forEach((sample,i)=>{
        if(shown!==index){markers[i].image.setAttribute('href',sample.image);markers[i].image.setAttribute('aria-label',sample.label);}
        const x=sample.start[0]+ease*(sample.end[0]-sample.start[0]);
        const y=sample.start[1]+ease*(sample.end[1]-sample.start[1]);
        markers[i].group.setAttribute('transform',`translate(${120+(x-cx)*scale-12},${98-(y-cy)*scale-12})`);
      });
      if(shown!==index)svg.setAttribute('aria-label',`${batch.dataset}: five colored classes with one hundred test samples each and representative images at class means, moving from input PCA to autoencoder bottleneck PCA. Overlap shows class mixing.`);
      shown=index;root.dataset.batch=String(index);root.dataset.progress=t.toFixed(3);
    }
    let elapsed=0,last=null,raf=0,paused=false,visible=true,disposed=false;
    const motion=matchMedia('(prefers-reduced-motion: reduce)');
    function preference(){draw(elapsed,motion.matches);}
    const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;});observer.observe(root);
    function tick(now){
      if(disposed)return;
      if(last!==null&&!paused&&visible&&!document.hidden&&!motion.matches){elapsed+=Math.min(now-last,100);draw(elapsed);}
      last=now;raf=requestAnimationFrame(tick);
    }
    motion.addEventListener('change',preference);preference();raf=requestAnimationFrame(tick);
    return {duration,batchDuration,seek(ms){elapsed=ms;paused=true;draw(ms);},pause(){paused=true;},play(){paused=false;},destroy(){disposed=true;cancelAnimationFrame(raf);observer.disconnect();motion.removeEventListener('change',preference);heading.remove();svg.remove();}};
  };
})();
