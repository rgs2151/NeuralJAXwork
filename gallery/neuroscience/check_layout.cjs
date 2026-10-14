/* Focused browser regression check against the approved gallery. */
const path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
 try{
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  for(const width of [1248,624]){
   await page.setViewportSize({width,height:300});const layouts=[];
   for(const file of ['../composite/index.html','index.html']){
    await page.goto('file://'+path.resolve(__dirname,file));
    await page.evaluate(()=>{gallery.seek(14000)});
    await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
    layouts.push(await page.evaluate(()=>[...document.querySelectorAll('[data-gallery-panel]')].map(p=>{
     const measure=e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height].map(v=>Math.round(v*1000)/1000)};
     return {panel:measure(p),heading:measure(p.firstElementChild),plot:measure(p.querySelector('svg')),font:getComputedStyle(p.firstElementChild).font};
    })));
   }
   assert.deepEqual(layouts[1],layouts[0]);
   const box=await page.evaluate(()=>{const r=document.querySelector('[data-gallery-panel="cnn"] svg text').getBBox();return {x:r.x,y:r.y,width:r.width,height:r.height}});
   assert(box.x>=0 && box.x+box.width<=280 && box.y>=0 && box.y+box.height<=280);
  }
  assert.deepEqual(errors,[]);console.log('MRI and approved gallery geometry match at full and half width; prediction text fits.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
