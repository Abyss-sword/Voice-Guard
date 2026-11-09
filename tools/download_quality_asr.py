"""Download one declared supplementary ASR model without changing old evaluation."""
from pathlib import Path
import json,hashlib
import whisper
model='small.en'
whisper.load_model(model,device='cpu')
path=Path.home()/'.cache/whisper'/f'{model}.pt'
out=Path('outputs/paper_quality/whisper-small-resource.json')
out.write_text(json.dumps({'model':model,'url':whisper._MODELS[model],
    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
    'purpose':'Supplementary clean-conversion calibration; historical base results retained'},indent=2)+'\n')
print(out)
