"""Download the upstream VQMIVC v0.0 inference assets into an isolated directory."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import time


def download_ranges(url, temporary, size):
    chunk_bytes=4*1024*1024
    parts=temporary.with_suffix('.parts')
    parts.mkdir(exist_ok=True)
    # Resolve the public release redirect once. Repeated GitHub redirects were
    # timing out while the asset CDN itself supports resumable byte ranges.
    request=urllib.request.Request(url+'?download=1',headers={'Range':'bytes=0-31'})
    with urllib.request.urlopen(request,timeout=60) as response:
        asset_url=response.geturl()
        response.read(32)
    def chunk(begin):
        end=min(begin+chunk_bytes,size)-1
        path=parts/str(begin)
        if path.exists() and path.stat().st_size==end-begin+1: return path
        for attempt in range(5):
            try:
                req=urllib.request.Request(asset_url+'&voiceguard-range='+str(begin),
                    headers={'Range':f'bytes={begin}-{end}'})
                with urllib.request.urlopen(req,timeout=120) as source:
                    if source.status!=206 or source.headers.get('Content-Range')!=f'bytes {begin}-{end}/{size}':
                        raise ValueError('Server did not honor the requested byte range')
                    data=source.read()
                if len(data)!=end-begin+1: raise ValueError('Truncated download range')
                path.write_bytes(data)
                print('Verified range',begin,end,flush=True)
                return path
            except Exception:
                if attempt==4: raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths=list(pool.map(chunk,range(0,size,chunk_bytes)))
    with temporary.open('wb') as out:
        for path in paths: out.write(path.read_bytes())


def main():
    root=Path('checkpoints/vqmivc-official')
    root.mkdir(parents=True,exist_ok=True)
    sources={'VQMIVC-model.ckpt-500.pt':('VQMIVC-model.ckpt-500.pt',112645564),
        'vocoder/checkpoint-3000000steps.pkl':('checkpoint-3000000steps.pkl',225316490),
        'vocoder/config.yml':('config.yml',2360),'vocoder/stats.h5':('stats.h5',4736)}
    receipt=root/'download-receipt.json'
    if receipt.exists(): raise ValueError('Resources already downloaded; preserve receipt')
    def fetch(item):
        target,(name,expected_size)=item
        path=root/target
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.partial')
        url='https://github.com/Wendison/VQMIVC/releases/download/v0.0/'+name
        if path.exists():
            if path.stat().st_size!=expected_size: raise ValueError('Existing asset has wrong size: '+str(path))
        else:
            if expected_size>1024*1024:
                download_ranges(url,temporary,expected_size)
                temporary.rename(path)
            for attempt in range(3):
                if path.exists(): break
                try:
                    size=0
                    request_url=url+'?download=1&voiceguard=full-'+str(attempt)
                    with urllib.request.urlopen(request_url,timeout=90) as source, temporary.open('wb') as out:
                        for chunk in iter(lambda:source.read(1024*1024),b''):
                            out.write(chunk);size+=len(chunk)
                    if size!=expected_size: raise ValueError('Upstream release size mismatch')
                    temporary.rename(path)
                    break
                except Exception:
                    if attempt==2: raise
                    time.sleep(2)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        size=path.stat().st_size
        print('Verified asset',name,size,flush=True)
        return target,{'sha256':h.hexdigest(),'bytes':size,'url':url,
            'integrity_note':'Locally recorded download digest; no independently published upstream digest established.'}
    with ThreadPoolExecutor(max_workers=4) as pool: values=dict(pool.map(fetch,sources.items()))
    receipt.write_text(json.dumps({'downloaded_utc':datetime.now(timezone.utc).isoformat(),
        'source_release':'https://github.com/Wendison/VQMIVC/releases/tag/v0.0','files':values},indent=2)+'\n')
    print(receipt,flush=True)


if __name__=='__main__': main()
