// Real browser upload -> Huawei GPU API -> actual result previews/download.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const runtime=process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
if(!runtime)throw new Error('Set CODEX_PRIMARY_RUNTIME_NODE_MODULES');
const {chromium}=await import(pathToFileURL(path.join(runtime,'playwright/index.mjs')).href);
const out=path.resolve('outputs/course/browser-live');await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({executablePath:process.env.EDGE_EXECUTABLE||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1080}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const started=Date.now();
try{
 await page.goto('http://127.0.0.1:5173/#/methods/group_xx_voice_guard',{waitUntil:'networkidle'});
 await page.getByRole('heading',{name:'Voice Guard 语音隐私主动防御',exact:true}).waitFor();
 // One required input plus an optional reference should still allow the sample button.
 await page.getByRole('button',{name:'用样例试一试',exact:true}).click();
 await page.locator('.upload_area.has_file').waitFor();
 await page.locator('#parameter_fast_mode').check();
 await page.locator('#parameter_method').selectOption('vg-projected');
 await page.locator('#parameter_target_speaker').fill('p225');
 await page.locator('#workspace').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(out,'01-workspace.png')});
 const responsePromise=page.waitForResponse(r=>r.url()==='http://127.0.0.1:8020/process'&&r.request().method()==='POST',{timeout:600000});
 await page.getByRole('button',{name:'开始处理',exact:true}).click();
 const response=await responsePromise;
 if(!response.ok())throw new Error(`GPU API returned ${response.status()}: ${await response.text()}`);
 const json=await response.json();
 await page.locator('.result_panel').waitFor({timeout:30000});
 await page.locator('.result_panel').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(out,'02-results.png')});
 if(await page.locator('.result_panel .artifact').count()!==6)throw new Error('Expected six rendered result artifacts');
 const waves=await page.locator('.result_panel audio').evaluateAll(async nodes=>{
  await Promise.all(nodes.map(a=>a.readyState>=1?null:new Promise((resolve,reject)=>{a.addEventListener('loadedmetadata',resolve,{once:true});a.addEventListener('error',()=>reject(new Error('Audio metadata failed')),{once:true});setTimeout(()=>reject(new Error('Audio metadata timeout')),20000);})));
  return nodes.map(a=>({src:a.currentSrc,duration:a.duration,readyState:a.readyState,error:a.error?.message||null}));
 });
 if(waves.length!==5||waves.some(a=>!Number.isFinite(a.duration)||a.duration<=0||a.error))throw new Error('Invalid playable audio results');
 const downloadPromise=page.waitForEvent('download');
 await page.locator('.result_panel .artifact').nth(1).getByRole('button',{name:'下载结果',exact:true}).click();
 const download=await downloadPromise;const downloaded=path.join(out,download.suggestedFilename());await download.saveAs(downloaded);
 const bytes=await fs.readFile(downloaded);
 if(!bytes.length)throw new Error('Empty browser download');
 await fs.writeFile(path.join(out,'response.json'),JSON.stringify(json,null,2));
 await fs.writeFile(path.join(out,'verification.json'),JSON.stringify({verified_utc:new Date().toISOString(),browser:await browser.version(),headless:true,
  real_browser_upload:true,real_gpu_response_status:response.status(),sample_with_optional_reference_loaded:true,
  rendered_artifact_count:6,playable_audio_metadata:waves,download:{filename:download.suggestedFilename(),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')},
  identity_verdict:json.meta.identity_verdict,quality_evaluated:json.meta.quality_evaluated,elapsed_s:(Date.now()-started)/1000,page_errors:errors},null,2));
 if(errors.length)throw new Error('Browser page errors: '+errors.join('; '));
 console.log('Browser upload, six rendered artifacts, five playable audio previews and actual download passed');
}finally{await browser.close();}
