"""Real Gradio GPU request, optional context restoration, and finite artifact checks."""
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:7860')
    parser.add_argument('--audio',type=Path,default=Path('data/VCTK22/p225/p225_040.wav'))
    parser.add_argument('--out',type=Path,default=Path('outputs/course/gradio-live'))
    args=parser.parse_args()
    from gradio_client import Client,handle_file
    import soundfile as sf
    import numpy as np
    import shutil
    client=Client(args.base,verbose=False)
    args.out.mkdir(parents=True,exist_ok=True)
    info=client.view_api(print_info=False,return_format='dict')
    write_json(args.out/'api.json',info)
    start=time.monotonic()
    result=client.predict(handle_file(str(args.audio.resolve())),'p225',None,.05,True,'vg',True,api_name='/on_protect')
    records=[]
    for name,source in zip(('original','defended','delta','converted_clean','converted_defended'),result[1:6]):
        dest=args.out/f'{name}.wav';shutil.copy2(source,dest)
        wave,sr=sf.read(dest,dtype='float32')
        if not wave.size or not np.isfinite(wave).all():raise ValueError('Invalid downloaded audio')
        records.append({'name':name,'sample_rate':sr,'samples':len(wave),'sha256':file_sha256(dest)})
    if len(records)!=5:raise ValueError('Expected five audio outputs')
    from vguard.audio import load_audio
    if len(load_audio(str(args.audio),records[0]['sample_rate']))!=records[0]['samples']:
        raise ValueError('Selected context restoration did not preserve full input length')
    shutil.copy2(result[6],args.out/'waves.png')
    write_json(args.out/'verification.json',{'model_backed_gradio_request_passed':True,'method':'vg',
        'preserve_context':True,'fast_mode':True,'elapsed_s':round(time.monotonic()-start,2),
        'status_html':result[0],'waves':records,'input_sha256':file_sha256(args.audio),
        'note':'Original method, full-context fast GPU demonstration, not defense/quality rate evaluation'})
    print('Gradio real request, five finite downloads and full context length passed')


if __name__=='__main__':main()
