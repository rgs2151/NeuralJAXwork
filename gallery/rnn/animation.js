/* Weather history rotates clockwise from a fixed present. Neurons are schematic. */
(function () {
  const NS = 'http://www.w3.org/2000/svg', RED = '#8b1e1e';
  const STROKE_WIDTH = 2;
  window.mountRNN = function (root, data) {
    const heading = document.createElement('div');
    heading.textContent = 'nj.Module.RNN';
    heading.style.cssText = 'font:700 18px/1.2 Arial,sans-serif;text-align:center;margin-bottom:2px;color:inherit';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '32 43 254 214');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'RNN weather clock: forecast temperatures in dark red and observed temperatures in black. Now is at three o’clock; clockwise positions show up to twenty-four days of history. Center neurons are illustrative.');
    svg.style.cssText = 'display:block;width:100%;height:auto;background:#fff;font-family:Arial,sans-serif';
    root.append(heading,svg);
    const plot=document.createElementNS(NS,'g');
    plot.setAttribute('transform','rotate(90 150 150)');
    svg.append(plot);
    function el(tag, attrs, parent = plot) {
      const node = document.createElementNS(NS, tag);
      for (const [k,v] of Object.entries(attrs)) node.setAttribute(k,v);
      if (attrs.stroke && attrs.stroke !== 'none') {
        node.setAttribute('stroke-width', STROKE_WIDTH);
        node.setAttribute('vector-effect', 'non-scaling-stroke');
        node.setAttribute('stroke-linecap', attrs['stroke-linecap'] || 'round');
        node.setAttribute('stroke-linejoin', 'round');
      }
      parent.append(node); return node;
    }
    function label(text,x,y,attrs={}) {
      const node=el('text',{x,y,fill:'#333','font-size':18,'font-weight':400,'text-anchor':'middle',...attrs});
      node.textContent=text;return node;
    }
    const cx=150,cy=150;
    const all=[...data.observed,...data.forecast];
    const min=Math.min(...all),max=Math.max(...all);
    const radius=value=>66+(value-min)/(max-min)*36.5;
    const polar=(angle,r)=>[cx+Math.sin(angle)*r,cy-Math.cos(angle)*r];
    // Short joined sections approximate a linear fade by historical age.
    const FADE_SECTIONS=96;
    function trace(name,color) {
      const group=el('g',{'data-trace':name});
      return Array.from({length:FADE_SECTIONS},(_,i)=>el('path',{
        fill:'none',stroke:color,opacity:1-i/(FADE_SECTIONS-1),'stroke-linecap':'butt',
        'data-age-fraction':i/(FADE_SECTIONS-1),
      },group));
    }
    const real=trace('real','#171717'),predicted=trace('forecast',RED);
    el('circle',{cx,cy,r:66,fill:'#fff',stroke:'#171717'});
    const locations=[[-20,-47],[18,-45],[44,-28],[-45,-20],[-13,-14],[20,-16],[45,9],[-48,12],[-22,24],[8,18],[35,37],[-11,48],[15,48],[0,-57]];
    const points=locations.map(([x,y])=>[x+cx,y+cy]);
    const connections=[[0,4],[4,6],[6,11],[11,3],[3,9],[9,1],[1,7],[7,10],[10,2],[2,5],[5,8],[8,12],[12,0],[0,10],[3,5],[4,12],[6,1],[7,4],[8,2],[9,11],[1,13],[13,4]];
    connections.forEach(([a,b],i)=>{
      const [x,y]=points[a],[xx,yy]=points[b];
      const bend=(i%2?1:-1)*(16+i%4*7);
      el('path',{d:`M${x} ${y}Q${(x+xx)/2+bend} ${(y+yy)/2-bend} ${xx} ${yy}`,fill:'none',stroke:'#4a2424'});
    });
    const nodes=points.map(([cx,cy])=>el('circle',{cx,cy,r:5.5,fill:'#fff',stroke:'#542323'}));
    // A schematic output connection stays anchored to the present, not to the rotating history.
    const output=el('path',{fill:'none',stroke:RED});
    // Counter-rotate the label so it reads horizontally above the output line.
    label('now',142,66,{transform:'rotate(-90 142 66)'});
    const observedNow=el('circle',{cx:150,r:3.4,fill:'#171717',stroke:'#fff'});
    const forecastNow=el('circle',{cx:150,r:3.8,fill:RED,stroke:'#fff'});
    const legend=document.createElement('div');
    legend.style.cssText='display:flex;justify-content:center;gap:14px;margin-top:-3px;font:18px Arial,sans-serif;color:inherit';
    for(const [color,name] of [[RED,'Forecast'],['#171717','Real']]) {
      const item=document.createElement('span');item.style.cssText='display:inline-flex;align-items:center;gap:5px';
      const dot=document.createElement('span');dot.style.cssText=`display:inline-block;width:8px;height:8px;border-radius:50%;background:${color}`;
      item.append(dot,document.createTextNode(name));legend.append(item);
    }
    root.append(legend);
    const interpolate=(series,t)=>{
      const i=Math.floor(t),f=t-i;return series[i]*(1-f)+series[Math.min(i+1,series.length-1)]*f;
    };
    const duration=18000;
    function draw(ms) {
      const hour=data.historyHours+(Math.max(0,ms)%duration)/duration*data.playbackHours;
      root.dataset.hour=hour.toFixed(3);
      function curve(series,paths) {
        // The oldest history stays separate from the current sample at the seam.
        paths.forEach((path,section)=>{
          const points=[];
          for(let j=0;j<=12;j++) {
            const age=(section+j/12)/FADE_SECTIONS*(data.historyHours-0.5);
            const [x,y]=polar(age/data.historyHours*Math.PI*2,radius(interpolate(series,hour-age)));
            points.push(`${j?'L':'M'}${x.toFixed(2)} ${y.toFixed(2)}`);
          }
          path.setAttribute('d',points.join(''));
        });
      }
      curve(data.observed,real);curve(data.forecast,predicted);
      const f=interpolate(data.forecast,hour),o=interpolate(data.observed,hour);
      const y=cy-radius(f);observedNow.setAttribute('cy',cy-radius(o));forecastNow.setAttribute('cy',y);
      output.setAttribute('d',`M150 93L150 ${y}`);
      nodes.forEach((node,i)=>{
        const seconds=Math.max(0,ms)%duration/1000;
        const activity=Math.pow((1+Math.sin(seconds*6+i*1.7)*Math.cos(seconds*2.3+i*0.43))/2,1.1);
        const rgb=[255,255,255].map((v,j)=>Math.round(v+([139,30,30][j]-v)*activity));
        node.setAttribute('fill',`rgb(${rgb.join(',')})`);
      });
    }
    const resize=new ResizeObserver(()=>{
      const size=18*254/(svg.getBoundingClientRect().width||300);
      svg.querySelectorAll('text').forEach(node=>node.setAttribute('font-size',size));
    });
    resize.observe(svg);
    let elapsed=0,last=null,raf=0,paused=false,visible=true,disposed=false;
    const motion=matchMedia('(prefers-reduced-motion: reduce)');
    function preference(){draw(motion.matches?6000:elapsed);}
    const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;});observer.observe(root);
    function tick(now){if(disposed)return;if(last!==null&&!paused&&visible&&!document.hidden&&!motion.matches){elapsed+=Math.min(now-last,100);draw(elapsed);}last=now;raf=requestAnimationFrame(tick);}
    motion.addEventListener('change',preference);preference();raf=requestAnimationFrame(tick);
    return {duration,seek(ms){elapsed=ms;paused=true;draw(ms);},pause(){paused=true;},play(){paused=false;},destroy(){disposed=true;cancelAnimationFrame(raf);observer.disconnect();resize.disconnect();motion.removeEventListener('change',preference);heading.remove();svg.remove();legend.remove();}};
  };
})();
