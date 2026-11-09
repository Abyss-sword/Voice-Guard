"""Isolate appendix-vs-source masking constants without changing masker selection."""
import json
from pathlib import Path
import sys
import librosa
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.masking import Bark, quiet, generate_th
from vguard.run_state import file_sha256, write_json


def constant_sensitivity(theta, sample_rate, n_fft):
    frequencies=librosa.fft_frequencies(sr=sample_rate,n_fft=n_fft)
    ath=np.full(len(frequencies),-np.inf)
    start=np.argmax(Bark(frequencies)>1)
    ath[start:]=quiet(frequencies[start:])
    total=np.power(10.,np.asarray(theta)/10)
    floor=np.power(10.,ath/10)[None]
    tonal=np.maximum(total-floor,0.)
    changed=10*np.log10(tonal*10**(5.4225/10)+floor*10**(12/10))
    return changed


def main():
    destination=Path('outputs/discriminant/masking-constant-diagnosis.json')
    if destination.exists(): raise ValueError('Preserve earlier diagnosis')
    protocol=json.loads(Path('docs/evidence/2026-11-09-content-robust/protocol.json').read_text())
    pp={'sample_rate':24000,'n_fft':2048,'hop_length':300,'win_length':1200}
    rows=[]
    for u in protocol['development']:
        path=Path('data/VCTK22')/u.split('_')[0]/(u+'.wav')
        wave,_=librosa.load(path,sr=pp['sample_rate'])
        crop,_=librosa.effects.trim(wave,top_db=15)
        theta,_=generate_th(crop,**pp)
        changed=constant_sensitivity(theta,pp['sample_rate'],pp['n_fft'])
        diff=changed-theta
        rows.append({'utterance':u,'input_sha256':file_sha256(path),
            'delta_min_db':float(diff.min()),'delta_mean_db':float(diff.mean()),
            'delta_max_db':float(diff.max()),'median_amplitude_cap_ratio':float(np.median(10**(diff/20)))})
    write_json(destination,{'diagnostic_only':True,'rows':rows,
        'paper_eq17_tonal_constant':-.6025,'upstream_tonal_constant':-6.025,
        'paper_eq13_quiet_offset_db':0,'upstream_quiet_offset_db':-12,
        'paper_pdf_sha256':file_sha256('refs/voiceguard_ijcai23.pdf'),
        'upstream_source_sha256':file_sha256('third_party/Voice-Guard/generate_masking_threshold.py'),
        'project_masking_source_sha256':file_sha256('vguard/masking.py'),
        'analysis_source_sha256':file_sha256(__file__),
        'interpretation':'Formula constants conflict with released source. Isolates the two constants using identical original masker selection and Bark cutoff; not a full alternate appendix implementation, audibility validation, or evidence identifying which convention generated the paper results.'})
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
