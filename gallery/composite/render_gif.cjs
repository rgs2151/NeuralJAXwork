/* Deterministic browser capture. Requires Playwright, Chromium and ffmpeg. */
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {spawnSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'neuraljaxlib-gallery-'));
 const fps=15,seconds=18,frames=Math.round(fps*seconds);
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1149,height:240},deviceScaleFactor:1});
  page.on('pageerror',e=>{throw e;});
  await page.goto('file://'+path.join(__dirname,'index.html'));
  for(let frame=0;frame<frames;frame++){
   await page.evaluate(ms=>gallery.seek(ms),frame*1000/fps);
   await page.screenshot({path:path.join(dir,String(frame).padStart(4,'0')+'.png')});
   if(frame%100===0)console.log(`Captured ${frame}/${frames}`);
  }
 }finally{await browser.close();}
 // Blend only the GIF's final half-second into its opening frame.
 // Native assets retain their independent clocks and exact original playback.
 const python=process.env.GALLERY_PYTHON||'python3';
 let result=spawnSync(python,['-c',`from PIL import Image
from pathlib import Path
p=Path(${JSON.stringify(dir)})
files=sorted(p.glob('*.png'))
first=Image.open(files[0]).convert('RGB')
for i,file in enumerate(files[-8:]):
    image=Image.open(file).convert('RGB')
    Image.blend(image,first,(i+1)/8).save(file)
`],{stdio:'inherit'});
 if(result.status!==0)throw new Error('Loop blend failed');
 result=spawnSync('ffmpeg',['-y','-loglevel','error','-framerate',String(fps),'-i',path.join(dir,'%04d.png'),'-filter_complex','split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle','-loop','0',path.join(__dirname,'gallery.gif')],{stdio:'inherit'});
 if(result.status!==0)throw new Error('GIF encoding failed');
 fs.rmSync(dir,{recursive:true});console.log('Saved gallery.gif');
})().catch(error=>{console.error(error);process.exitCode=1;});
