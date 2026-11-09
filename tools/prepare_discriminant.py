"""Freeze enrollment-native LDA, its development comparison and untouched sentences."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.discriminant_recovery import fit_discriminant, EnrollmentDiscriminant
from vguard.run_state import file_sha256, write_json
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize


def read(path): return json.loads(Path(path).read_text())


def main():
    evidence=Path('docs/evidence/2026-11-09-discriminant')
    if (evidence/'protocol.json').exists(): raise ValueError('Protocol already frozen')
    prior=read('docs/evidence/2026-11-09-content-robust/protocol.json')
    forbidden=set()
    for path in Path('docs/evidence').rglob('*protocol.json'):
        if path.as_posix().endswith('2026-11-09-discriminant/initial-protocol.json'):
            if Path('outputs/discriminant/2026-11-09/validation-state.json').exists():
                raise ValueError('Initial validation unexpectedly examined')
            continue
        value=read(path)
        for key in ('development','validation'):
            if isinstance(value.get(key),list): forbidden.update(x for x in value[key] if isinstance(x,str))
        forbidden.update(Path(x).stem for x in value.get('enrollment_sha256',{}))
    first=read('docs/evidence/2026-11-09/frozen-protocol.json')
    forbidden.update(r['utterance'] for r in first['new_utterances'])
    torch.set_num_threads(1)
    model,config,attr=load_adainvc('checkpoints/adain-vc-official','cuda')
    with np.load(prior['native_target_bank']) as p: names=list(p['names'])
    vectors=[]; labels=[]; enrollment={}
    with torch.inference_mode():
        for index,name in enumerate(names):
            for path in sorted((Path('data/VCTK22')/str(name)).glob('*.wav'))[:30]:
                mel,_=get_spectrograms(str(path),**config['preprocess'])
                emb=model.speaker_encoder(torch.tensor(normalize(mel,attr).T[None],device='cuda'))
                vectors.append(emb.squeeze().cpu().numpy()); labels.append(index)
                enrollment[path.as_posix()]=file_sha256(path)
            print('Enrollment',name,flush=True)
    if len(enrollment)!=600: raise ValueError('Need exactly 20 x 30 enrolled utterances')
    fit=fit_discriminant(vectors,labels,len(names),shrinkage=.2)
    bank=Path('outputs/discriminant/enrollment-lda-v2.npz')
    if bank.exists(): raise ValueError('Preserve enrollment bank')
    bank.parent.mkdir(parents=True,exist_ok=True)
    np.savez(bank,**fit,names=np.asarray(names),features=np.asarray(vectors),labels=np.asarray(labels))
    validation=[]
    for name in names[:10]:
        files=sorted((Path('data/VCTK22')/str(name)).glob('*.wav'))
        if len(files)<78: raise ValueError('Insufficient untouched sentences')
        validation.extend(p.stem for p in files[76:78])
    if set(validation)&forbidden or len(set(validation))!=20:
        raise ValueError('Validation overlaps any observed/enrollment sentence')
    assets={k:v for k,v in prior['assets'].items() if not k.startswith('data/')}
    assets[bank.as_posix()]=file_sha256(bank)
    for u in prior['development']+validation+prior['evaluation_contents']+['p225_002','p226_002']:
        for suffix in ('.wav','.txt'):
            p=Path('data/VCTK22')/u.split('_')[0]/(u+suffix); assets[p.as_posix()]=file_sha256(p)
    code=dict(prior['code_sha256'])
    for name in ('vguard/discriminant_recovery.py','scripts/run_discriminant.py',
            'tools/prepare_discriminant.py','tools/decide_discriminant_validation.py',
            'tests/test_discriminant_recovery.py'):
        code[name]=file_sha256(name)
    for name,digest in {**assets,**code}.items():
        if file_sha256(name)!=digest: raise ValueError('Frozen input changed: '+name)
    protocol={**prior,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'output':'outputs/discriminant/2026-11-09-v2','seed':2026110906,
        'type':'Enrollment-discriminant native/decoded identity margin with original CTC constraints',
        'validation':validation,'validation_examined':False,
        'validation_selection':'Sorted original sentence indices 76:78, first ten enrolled speakers; exclude all prior observed, optimized, validation and enrollment sentences.',
        'optimization_contents':['p225_002','p226_002'],
        'content_design':'Two fixed optimization contents; four pre-existing evaluation contents, including both optimization-overlap and held-out text. Not all conversion contents are held out.',
        'profiles':[{'name':'lda_margin_snr20','snr_db':20.,'mask_relax_db':0.},
            {'name':'lda_margin_snr18_mask3','snr_db':18.,'mask_relax_db':3.}],
        'refinement':{'steps':800,'learning_rate':.0003,'max_seconds':420.},
        'discriminant_bank':bank.as_posix(),'enrollment_sha256':enrollment,
        'discriminant_fit':{'classes':20,'utterances_per_class':30,'within_covariance_shrinkage':.2,
            'enrollment_accuracy':fit['enrollment_accuracy'],'training_margin_scale':fit['scale'],
            'note':'Training enrollment accuracy only, not independent generalization evidence.'},
        'gradient_design':'Shrinkage LDA fitted solely to native enrollment Es; equal class priors, centered means. Minimize smooth worst-case owner-vs-best-impostor score margin over reference and two decoded-mel re-encodings. Full-context CTC conflicts projected away; add CTC descent when above baseline. No Griffin-Lim or independent scorer gradient.',
        'proxy_acceptance':'Every 16 steps accept strictly improved training identity loss with CTC<=baseline+.01, original ECAPA cosine>=baseline-.005 and relative spectral distance<=baseline+.03. Reject/rollback and lower rate otherwise. Recheck actual PCM16 before independent evaluation; failure copies baseline.',
        'relaxed_budget_note':'The second profile permits SNR18 and +3dB masking allowance. It is a different perturbation budget and cannot establish same-distortion or inaudibility equivalence.',
        'limits':{'development_hours':1.,'validation_hours':1.25,'automatic_shutdown':False},
        'prior_protocol_sha256':file_sha256('docs/evidence/2026-11-09-content-robust/protocol.json'),
        'source_note':'New native discriminant extension, not an official Voice Guard or Kalypta implementation.',
        'assets':assets,'code_sha256':code,
        'limitations':['Development sentences already observed; fresh validation only if fixed development gate passes.',
            'LDA/native/CTC/ECAPA training proxies may overfit; GE2E, MOSNet and both Whisper models remain held out.',
            'One candidate relaxes the SNR and masking budget; that alone can change quality and defense.',
            'Twenty fresh sentences from ten enrolled speakers, not unseen-pretraining people.',
            'No human listening study, second VC model or paper-scale quality equivalence claim.']}
    # Drop stale fields from the preceding experiment rather than misdescribe execution.
    for key in ('smoke_sha256','eot_channels','environment_limits'):
        protocol.pop(key,None)
    protocol['implementation_correction']='Initial attempt interrupted during development after discovery that the reused projector rejects negative margins. Corrected allowance scales its positive amplitude cap instead. Initial protocol, source archive, partial WAVs, log and state retained. No initial validation was run; reserved validation sentences remain untouched.'
    write_json(evidence/'protocol.json',protocol)
    print('Frozen',evidence/'protocol.json',file_sha256(evidence/'protocol.json'),validation,flush=True)


if __name__=='__main__': main()
