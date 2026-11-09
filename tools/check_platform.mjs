// Validate this project's configuration against the actual course frontend schema.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const repo = path.resolve(process.argv[2] || 'outputs/course/platform-frontend');
const output = path.resolve(process.argv[3] || 'outputs/course/platform-contract.json');
const {validateManifest,validateParameters,defaultsFor} = await import(pathToFileURL(path.join(repo,'src/lib/contract.js')).href);
const configPath = path.join(repo,'public/methods/group_xx_voice_guard.json');
const raw = await fs.readFile(configPath);
const method = validateManifest(JSON.parse(raw.toString('utf8')));
const errors = validateParameters(method,defaultsFor(method));
if (Object.keys(errors).length) throw new Error(JSON.stringify(errors));
const assets=[];
for(const example of method.examples) {
  for (const artifact of [example.input,example.output]) {
    const file = path.resolve(repo,'public',artifact.src);
    const bytes = await fs.readFile(file);
    assets.push({path:artifact.src,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
  }
}
const result={checked_utc:new Date().toISOString(),method_id:method.id,configuration_schema_valid:true,
              default_parameters_valid:true,configuration_sha256:createHash('sha256').update(raw).digest('hex'),
              assets,endpoint:method.adapter.endpoint,live_api_tested:false,
              note:'Schema, defaults and preview files only. Live request and returned artifacts require a separate API test.'};
await fs.mkdir(path.dirname(output),{recursive:true});
await fs.writeFile(output,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
