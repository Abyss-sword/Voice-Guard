"""Exercise a real deployed adapter and download every returned artifact.

Run only against the local/SSH-forwarded course service with loaded real weights.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import urljoin,urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8020')
    parser.add_argument('--audio',type=Path,default=Path('data/VCTK22/p225/p225_040.wav'))
    parser.add_argument('--out',type=Path,default=Path('outputs/course/platform-live'))
    parser.add_argument('--method',choices=['vg','vg-projected'],default='vg-projected')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--native-upload',action='store_true',help='Use input bytes; default verifies stereo 22050-Hz upload')
    args=parser.parse_args()
    import requests
    import numpy as np
    import soundfile as sf
    import librosa
    args.out.mkdir(parents=True,exist_ok=True)
    health=requests.get(args.base+'/health',timeout=10)
    health.raise_for_status()
    if health.json()['status']!='ok':raise ValueError('Model not ready')
    upload=args.audio
    if not args.native_upload:
        wave,sr=sf.read(upload,dtype='float32')
        if wave.ndim>1:wave=wave.mean(axis=1)
        wave=librosa.resample(wave,orig_sr=sr,target_sr=22050)
        upload=args.out/'stereo-22050.wav'
        sf.write(upload,np.stack((wave,wave),axis=1),22050,subtype='PCM_16')
    parameters={'eps':.05,'fast_mode':not args.full,'target_speaker':'','method':args.method,'preserve_context':False}
    def send():
        with upload.open('rb') as stream:
            return requests.post(args.base+'/process',data={'parameters':json.dumps(parameters),'method_id':'group_xx_voice_guard'},
                                 files={'defend_audio':(upload.name,stream,'audio/wav')},timeout=2400)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(send)
        # Sample once while the real GPU request is active; bounded, no busy polling.
        time.sleep(2)
        health_start=time.monotonic()
        during=requests.get(args.base+'/health',timeout=10)
        during.raise_for_status()
        health_elapsed=time.monotonic()-health_start
        response=future.result()
    response.raise_for_status()
    result=response.json()
    write_json(args.out/'response.json',result)
    artifacts=result['artifacts']
    if len(artifacts)!=6:raise ValueError(f'Expected six artifacts, got {len(artifacts)}')
    downloads=[]
    for item in artifacts:
        name=Path(item['filename']).name
        if name!=item['filename']:raise ValueError('Unexpected artifact filename')
        url=urljoin(args.base+'/',item['url'])
        if urlsplit(url).netloc != urlsplit(args.base).netloc:raise ValueError('Artifact host differs from requested service')
        content=requests.get(url,timeout=60);content.raise_for_status()
        dest=args.out/name;dest.write_bytes(content.content)
        downloads.append({'name':name,'bytes':len(content.content),'sha256':file_sha256(dest),'status':content.status_code})
        if item['kind']=='audio':
            wave,sr=sf.read(dest,dtype='float32')
            if wave.size==0 or not np.isfinite(wave).all():raise ValueError('Invalid audio download')
    if result['meta'].get('quality_evaluated') is not False:raise ValueError('Adapter must disclose missing quality scores')
    write_json(args.out/'verification.json',{'verified_utc':datetime.now(timezone.utc).isoformat(),
        'base':args.base,'input_sha256':file_sha256(args.audio),'upload_sha256':file_sha256(upload),
        'parameters':parameters,'elapsed_s':round(time.monotonic()-started,2),'downloads':downloads,
        'health_during_request_s':round(health_elapsed,3),'model_backed_http_request_passed':True,
        'identity_verdict':result['meta']['identity_verdict'],'quality_joint_claim':False})
    print('Real HTTP request and all six downloads passed:',args.out)


if __name__=='__main__':main()
