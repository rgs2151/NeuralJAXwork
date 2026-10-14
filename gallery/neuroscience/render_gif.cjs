/* Render the separate MRI gallery at the approved composite's exact dimensions. */
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {spawnSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'neuraljaxlib-mri-'));
 const only=process.argv.includes('--composite-only');
 const fps=15,frames=Math.ceil((only?18:43.5)*fps);
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1248,height:270},deviceScaleFactor:1});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('file://'+path.join(__dirname,'index.html'));
  const geometry=await page.evaluate(()=>({width:gallery.width,height:gallery.height,panels:[...document.querySelectorAll('[data-gallery-panel]')].map(p=>({name:p.dataset.galleryPanel,width:p.getBoundingClientRect().width,font:getComputedStyle(p.firstElementChild).fontSize}))}));
  if(geometry.width!==1248||geometry.height!==270||geometry.panels.some(p=>p.width!==240||p.font!=='18px'))throw Error('Gallery geometry changed');
  fs.writeFileSync(path.join(__dirname,'geometry.json'),JSON.stringify(geometry,null,2)+'\n');
  for(let frame=0;frame<frames;frame++){
   await page.evaluate(ms=>gallery.seek(ms),frame*1000/fps);
   await page.screenshot({path:path.join(dir,String(frame).padStart(4,'0')+'.png')});
   if(frame%150===0)console.log(`Captured ${frame}/${frames}`);
  }
  if(errors.length)throw Error(errors.join('\n'));
 }finally{await browser.close();}
 const jobs=[['gallery.gif',18,'null'],['../cnn_brain_mri/animation.gif',43.5,'crop=240:270:0:0'],['../gan_brain_mri/animation.gif',4.59,'crop=240:270:756:0']];
 for(const [file,seconds,crop] of jobs.filter(job=>!only||job[0]==='gallery.gif')){
  const result=spawnSync('ffmpeg',['-y','-loglevel','error','-framerate',String(fps),'-i',path.join(dir,'%04d.png'),'-t',String(seconds),'-filter_complex',`${crop},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle`,'-loop','0',path.join(__dirname,file)],{stdio:'inherit'});
  if(result.status!==0)throw Error('GIF encoding failed');
 }
 const finish=spawnSync(process.env.GALLERY_PYTHON||'python3',[path.join(__dirname,'finish_gif.py')],{stdio:'inherit'});
 if(finish.status!==0)throw Error('Composite loop blend failed');
 fs.rmSync(dir,{recursive:true});console.log('Saved three MRI GIFs.');
})().catch(error=>{console.error(error);process.exitCode=1;});
