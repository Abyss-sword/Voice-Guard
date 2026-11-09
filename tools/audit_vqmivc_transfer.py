"""Verify fixed transfer pairings, independent gates and recorded WAV bytes."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.metrics.staged import gates
from vguard.run_state import file_sha256,utterance_seed,write_json
from tools.analyze_staged_recovery import cluster_intervals,rate


def summarize(rows,asv='ge2e',asr='small_en'):
    key='gates_'+asv;unique={r['utterance']:r for r in rows}
    identity=sum(r[key]['clean_identity_ok'] for r in rows)
    full=sum(r[key]['clean_full_quality_ok'] for r in rows)
    return {'utterances':len(unique),'conversion_pairs':len(rows),
        'protected_quality_count':sum(r[key]['protected_quality_ok'] for r in unique.values()),
        'protected_identity_retained':sum(r['defended']['asv_'+asv]==u.split('_')[0] for u,r in unique.items()),
        'protected_mos_mean':statistics.mean(r['defended']['mos_original'] for r in unique.values()),
        'protected_wer_macro':statistics.mean(r['defended']['wer_'+asr] for r in unique.values()),
        'protected_cer_macro':statistics.mean(r['defended']['cer_'+asr] for r in unique.values()),
        'clean_converted_mos_mean':statistics.mean(r['clean_vc']['mos_original'] for r in rows),
        'clean_converted_wer_macro':statistics.mean(r['clean_vc']['wer_'+asr] for r in rows),
        'protected_converted_mos_mean':statistics.mean(r['protected_vc']['mos_original'] for r in rows),
        'protected_converted_wer_macro':statistics.mean(r['protected_vc']['wer_'+asr] for r in rows),
        'converted_identity_retained':sum(r['protected_vc']['asv_'+asv]==r['utterance'].split('_')[0] for r in rows),
        'strict_joint':rate(sum(r[key]['strict_joint'] for r in rows),identity),
        'paper_identity_screen':rate(sum(r[key]['paper_identity_screen'] for r in rows),identity),
        'paper_full_screen':rate(sum(r[key]['paper_full_screen'] for r in rows),full)}


def main():
    evidence=Path('docs/evidence/2026-11-09-vqmivc')
    read=lambda p:json.loads(Path(p).read_text())
    p=read(evidence/'protocol.json');digest=file_sha256(evidence/'protocol.json');out=Path(p['output'])
    state=read(out/'state.json');rows=read(out/'results.json')['rows']
    expected={(u,c,n) for u in p['utterances'] for c in p['contents'] for n in p['profiles']}
    if state['state']!='complete' or state['protocol_sha256']!=digest or state['rows']!=len(expected):
        raise ValueError('Incomplete transfer')
    if len(rows)!=len(expected) or {(r['utterance'],r['content'],r['profile']) for r in rows}!=expected:
        raise ValueError('Missing/duplicate transfer rows')
    for name,sha in {**p['assets'],**p['code_sha256']}.items():
        if file_sha256(name)!=sha:raise ValueError('Frozen input changed: '+name)
    measured={};controls={};originals={}
    for r in rows:
        u,c,n=r['utterance'],r['content'],r['profile'];seed=utterance_seed(p['seed'],u+':'+c)
        if r['protocol_sha256']!=digest or r['conversion']['seed']!=seed or r['clean_conversion']['seed']!=seed:
            raise ValueError('Protocol or paired noise changed')
        clean={k:r[k] for k in ('original','clean_vc','clean_conversion')}
        if (u,c) in controls and controls[u,c]!=clean:raise ValueError('Unpaired controls')
        controls[u,c]=clean
        observed={k:r[k] for k in ('original','defended')}
        if (u,n) in originals and originals[u,n]!=observed:raise ValueError('Repeated originals differ')
        originals[u,n]=observed
        paths={'original':Path(p['inputs'][u]['original']),
            'defended':Path(p['inputs'][u]['defended'][n]),
            'clean_vc':out/'controls'/u/(c+'.wav'),'protected_vc':out/n/u/(c+'.wav')}
        for key,path in paths.items():
            if file_sha256(path)!=r[key]['sha256']:raise ValueError('Scored WAV changed: '+str(path))
            measured[path.as_posix()]=r[key]['sha256']
            wave,sr=sf.read(path,dtype='float32')
            if wave.ndim!=1 or not np.isfinite(wave).all() or np.std(wave)<1e-5:
                raise ValueError('Invalid waveform')
            if key in ('clean_vc','protected_vc') and sr!=16000:raise ValueError('Unexpected conversion sample rate')
        for asv,asr in (('ge2e','small_en'),('ecapa','base')):
            if r['gates_'+asv]!=gates(u,r['original'],r['clean_vc'],r['defended'],r['protected_vc'],asv=asv,asr=asr):
                raise ValueError('Gates do not recompute')
    groups={n:[r for r in rows if r['profile']==n] for n in p['profiles']}
    baseline=groups[p['profiles'][0]]
    analysis={'counts_verified':True,'protocol_sha256':digest,'analysis_code_sha256':file_sha256(__file__),
        'results_sha256':file_sha256(out/'results.json'),'rows':len(rows),'measured_wavs':len(measured),
        'profiles':{n:{'primary':summarize(rs),'secondary':summarize(rs,'ecapa','base'),
            'by_content':{c:summarize([r for r in rs if r['content']==c]) for c in p['contents']}} for n,rs in groups.items()},
        'paired_intervals':{n:cluster_intervals(baseline,rs,'gates_ge2e') for n,rs in groups.items() if n!=p['profiles'][0]},
        'conversion_clipping':{n:{'max_abs_peak':max(r['conversion']['abs_peak_before_pcm'] for r in rs),
            'max_clipped_fraction':max(r['conversion']['clipped_fraction_before_pcm'] for r in rs)} for n,rs in groups.items()},
        'clean_control_failures':dict(Counter('passes_full' if r['gates_ge2e']['clean_full_quality_ok'] else
            'identity_fails' if not r['gates_ge2e']['clean_identity_ok'] else 'quality_fails' for r in baseline)),
        'limitations':p['limitations'],'no_method_selection':True,'unseen_sentence_validation':False}
    destination=evidence/'analysis.json'
    if destination.exists() and read(destination)!=analysis:raise ValueError('Preserve previous transfer audit')
    write_json(destination,analysis)
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
