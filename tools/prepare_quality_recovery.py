"""Freeze recovery profiles, data and selection rule before running new experiments."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    out=Path('docs/evidence/2026-11-09-quality/recovery-protocol.json')
    if out.exists():raise ValueError('Protocol already exists; do not refreeze in place')
    development=['p225_040','p226_039','p227_033']
    validation=[]
    for speaker in ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234']:
        files=sorted((Path('data/VCTK22')/speaker).glob('*.wav'))
        selected=files[50:52] if len(files)>52 else files[30:31]
        if not selected:raise ValueError('Missing holdout speaker: '+speaker)
        validation.extend(path.stem for path in selected)
    old=json.loads(Path('docs/evidence/2026-11-09/frozen-protocol.json').read_text(encoding='utf-8'))
    if set(validation)&({row['utterance'] for row in old['new_utterances']}|set(development)):
        raise ValueError('Validation overlaps prior optimized utterances')
    contents=['p226_002','p225_002','p227_002']
    assets=[Path('data/VCTK22')/utterance.split('_')[0]/(utterance+'.'+suffix)
            for utterance in development+validation+contents for suffix in ('wav','txt')]
    assets += [Path(name) for name in ('outputs/registry.npz','outputs/registry_vc.npz',
        'checkpoints/adain-vc-official/vctk_model.ckpt','checkpoints/adain-vc-official/attr.pkl',
        'checkpoints/adain-vc-official/config.yaml','checkpoints/ecapa-local/embedding_model.ckpt',
        'checkpoints/mosnet-original/cnn_blstm.h5','checkpoints/mosnet-hf/pytorch_model.bin',
        'checkpoints/mosnet-hf/config.json','checkpoints/mosnet-hf/modeling_mosnet.py','checkpoints/mosnet-hf/configuration_mosnet.py')]
    code=['scripts/run_quality_recovery.py','vguard/recovery.py','vguard/features.py','vguard/masking.py',
          'vguard/audio.py','vguard/asv.py','vguard/vc/adainvc.py','vguard/metrics/mos_original.py',
          'vguard/metrics/mos.py','vguard/metrics/wer.py','third_party/ada_in_vc/model.py']
    profiles=[{'name':name,'objective':objective,'snr_db':snr,'margin_db':3.,'learning_rate':.0003}
              for name,objective,snr in [('embedding_snr25','embedding',25.),
                ('feedback_snr25','feedback',25.),('feedback_snr20','feedback',20.)]]
    write_json(out,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'type':'New recovery protocol after earlier negative results',
        'output':'outputs/paper_quality/2026-11-09-recovery','seed':20261109,
        'model_dir':'checkpoints/adain-vc-official','vctk':'data/VCTK22',
        'eval_registry':'outputs/registry.npz','attack_registry':'outputs/registry_vc.npz',
        'development':development,'validation':validation,'profiles':profiles,'checkpoints':[500,1500],
        'optimization_contents':contents[:2],
        'evaluation_contents':{'development':contents[:1],'validation':[contents[0],contents[2]]},
        'selection_rule':'Global profile/checkpoint maximizing strict identity joint count, full-control paper-structure joint count, protected quality count, then lower macro WER, higher mean original MOSNet, higher SNR budget, fewer steps, profile name. Development only. No per-utterance or validation selection.',
        'mos_evaluators':{'primary':'Original lochenchou/MOSNet CNN-BLSTM, cross-framework parity verified',
                          'secondary':'Historical HF model; retain separately'},
        'decoder':{'whisper':'base','temperature':0,'fp16':False,'language':'en'},
        'whisper_base_sha256':'ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e',
        'gates':{'mos_min':3.,'wer_max':.10,'strict':'Protected identity preserved and converted identity changed, clean identity correct',
            'paper_structure':'Protected identity and quality retained; converted identity OR MOS OR content fails. WER<=10% is an explicit operational interpretation of ambiguous ASR threshold.',
            'full_clean_screen':'Additionally require clean conversion MOS>=3 and WER<=10%, guaranteeing no quality disruption at zero noise'},
        'validation_note':'15 new utterances; p230-p234 not used in attack development but present in enrollment and potentially VC pretraining. Not a claim of unseen-pretraining-speaker generalization.',
        'audio':'Preserve full original context; perturb original top_db=15 crop only; PCM16 export; evaluate decoded export; no loudness normalization',
        'limits':{'development_hours':1,'validation_hours':1,'no_automatic_shutdown':True},
        'assets':{path.as_posix():file_sha256(path) for path in assets},
        'code_sha256':{name:file_sha256(name) for name in code}})
    print(out,file_sha256(out));print('Validation:',','.join(validation))


if __name__=='__main__':main()
