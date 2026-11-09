"""Bounded development and frozen holdout experiments for paper-quality recovery.

New protocol, original MOSNet and fixed greedy Whisper decoding. Historical scores
are never overwritten. No independent evaluator participates in the attack loss.
"""
from __future__ import annotations
import argparse,json,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vguard.audio import load_audio,restore_protected_crop
from vguard.asv import embed_wav,load_ecapa
from vguard.features import wav2mel_tensor
from vguard.masking import generate_th,perturbation_psd
from vguard.metrics.mos import load_mosnet,mos
from vguard.metrics.mos_original import OriginalMOSNet
from vguard.metrics.wer import load_whisper,wer,cer
from vguard.recovery import recover_waveform
from vguard.run_state import file_sha256,write_json,utterance_seed
from vguard.vc.adainvc import load_adainvc,convert,normalize,get_spectrograms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--split',choices=['development','validation'],required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--max-hours',type=float,default=2)
    args=parser.parse_args();started=time.monotonic()
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    for name,digest in protocol['assets'].items():
        if file_sha256(name)!=digest:raise ValueError('Protocol asset changed: '+name)
    for name,digest in protocol['code_sha256'].items():
        if file_sha256(name)!=digest:raise ValueError('Protocol code changed: '+name)
    out=Path(protocol['output']);out.mkdir(parents=True,exist_ok=True)
    profiles=protocol['profiles']
    if args.split=='validation':
        if not args.selection:parser.error('Freeze development selection before validation')
        selection=json.loads(args.selection.read_text(encoding='utf-8'))
        if selection['protocol_sha256']!=file_sha256(args.protocol):raise ValueError('Wrong selection protocol')
        profiles=[selection['profile']]
        if profiles[0] not in protocol['profiles']:raise ValueError('Unregistered profile')
        checkpoints=(selection['checkpoint'],)
    else:checkpoints=tuple(protocol['checkpoints'])
    state_path=out/(args.split+'-state.json')
    state={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),
           'protocol_sha256':file_sha256(args.protocol),'split':args.split,
           'selection_sha256':file_sha256(args.selection) if args.selection else None}
    write_json(state_path,state)
    model,config,attr=load_adainvc(protocol['model_dir'],'cuda')
    pp=config['preprocess'];sr=pp['sample_rate']
    kwargs={key:pp[key] for key in ('sample_rate','preemph','n_fft','hop_length','win_length','n_mels','ref_db','max_db','top_db')}
    independent=load_ecapa(device='cuda');original_mos=OriginalMOSNet(device='cuda')
    old_mos=load_mosnet(device='cuda');recognizer=load_whisper('base','cuda')
    with np.load(protocol['eval_registry'],allow_pickle=False) as registry:
        names=list(registry['names']);embeddings=np.array(registry['embeddings'])
    with np.load(protocol['attack_registry'],allow_pickle=False) as registry:
        attack_names=list(registry['names']);attack_embeddings=np.array(registry['embeddings'])
    content_mels=[]
    for utterance in protocol['optimization_contents']:
        path=Path(protocol['vctk'])/utterance.split('_')[0]/(utterance+'.wav')
        mel,_=get_spectrograms(str(path),**pp)
        content_mels.append(torch.from_numpy(normalize(mel,attr)).T.unsqueeze(0).cuda())
    def predict(wave):
        embedding=embed_wav(independent,wave,sr)
        sims=embeddings@embedding/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(embedding)+1e-9)
        return str(names[int(np.argmax(sims))]),{str(name):float(value) for name,value in zip(names,sims)}
    def score(path,reference):
        wave,rate=sf.read(path,dtype='float32')
        if rate!=sr:raise ValueError('Unexpected sample rate')
        hyp=recognizer.transcribe(str(path),language='en',temperature=0,fp16=False)['text'].strip()
        pred,sims=predict(wave)
        return {'asv':pred,'similarities':sims,'mos_original':original_mos.score(wave,sr),
                'mos_historical_model':mos(old_mos,wave,sr),'wer':wer(hyp,reference),
                'cer':cer(hyp,reference),'hypothesis':hyp,'reference':reference,'sha256':file_sha256(path)}
    rows=[]
    def bounded():
        if time.monotonic()-started>args.max_hours*3600:raise TimeoutError('Experiment wall-time cap reached')
    try:
        for utterance in protocol[args.split]:
            bounded();speaker=utterance.split('_')[0]
            path=Path(protocol['vctk'])/speaker/(utterance+'.wav')
            reference=path.with_suffix('.txt').read_text().strip()
            full=load_audio(str(path),sr);crop,interval=librosa.effects.trim(full,top_db=pp['top_db'])
            theta,psd_max=generate_th(crop,sr,pp['n_fft'],pp['hop_length'],pp['win_length'])
            own=attack_embeddings[attack_names.index(speaker)]
            sims=attack_embeddings@own/(np.linalg.norm(attack_embeddings,axis=1)*np.linalg.norm(own)+1e-9)
            target_index=int(np.argmin(sims));target_embedding=attack_embeddings[target_index]
            clean_dir=out/'controls'/utterance;clean_dir.mkdir(parents=True,exist_ok=True)
            clean_path=clean_dir/'original.wav'
            if not clean_path.exists():sf.write(clean_path,full,sr,subtype='PCM_16')
            contents=protocol['evaluation_contents'][args.split]
            controls={}
            for content in contents:
                control_path=clean_dir/(content+'.json')
                if control_path.exists():controls[content]=json.loads(control_path.read_text());continue
                source=Path(protocol['vctk'])/content.split('_')[0]/(content+'.wav')
                source_ref=source.with_suffix('.txt').read_text().strip()
                converted=clean_dir/(content+'.wav')
                seed=utterance_seed(protocol['seed'],utterance+':'+content)
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                convert(model,config,attr,str(source),str(clean_path),str(converted),'cuda')
                controls[content]={'original':score(clean_path,reference),'converted':score(converted,source_ref),
                                   'seed':seed,'utterance':utterance,'content':content}
                write_json(control_path,controls[content])
            for profile in profiles:
                bounded();name=profile['name'];job=out/args.split/name/utterance;job.mkdir(parents=True,exist_ok=True)
                optimization=job/'optimization.json'
                if not optimization.exists():
                    seed=utterance_seed(protocol['seed'],utterance+':'+name)
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    def save(step,wave):
                        destination=job/f'step_{step}';destination.mkdir(parents=True,exist_ok=True)
                        full_protected=restore_protected_crop(full,crop,wave,*map(int,interval))
                        sf.write(destination/'defended.wav',full_protected,sr,subtype='PCM_16')
                    _,trace=recover_waveform(model,crop,target_embedding,content_mels,theta,psd_max,attr,kwargs,
                        objective=profile['objective'],steps=max(checkpoints),learning_rate=profile['learning_rate'],
                        snr_db=profile['snr_db'],margin_db=profile['margin_db'],checkpoints=checkpoints,callback=save)
                    write_json(optimization,{'trace':trace,'profile':profile,'seed':seed,
                        'target_prototype':str(attack_names[target_index]),'crop_interval':list(map(int,interval)),
                        'evaluators_used_in_loss':False,'protocol_sha256':file_sha256(args.protocol),
                        'audio_sha256':{str(step):file_sha256(job/f'step_{step}/defended.wav') for step in checkpoints}})
                for checkpoint in checkpoints:
                    destination=job/f'step_{checkpoint}';protected=destination/'defended.wav'
                    saved=json.loads(optimization.read_text())
                    if saved['audio_sha256'][str(checkpoint)]!=file_sha256(protected):raise ValueError('Saved audio changed')
                    for content,control in controls.items():
                        record=destination/(content+'.json')
                        if record.exists():rows.append(json.loads(record.read_text()));continue
                        bounded();source=Path(protocol['vctk'])/content.split('_')[0]/(content+'.wav')
                        converted=destination/(content+'.wav')
                        seed=control['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        convert(model,config,attr,str(source),str(protected),str(converted),'cuda')
                        defended_score=score(protected,reference)
                        converted_score=score(converted,source.with_suffix('.txt').read_text().strip())
                        identity_clean=control['converted']['asv']==speaker
                        clean_quality=control['converted']['mos_original']>=3 and control['converted']['wer']<=.10
                        quality=defended_score['asv']==speaker and defended_score['mos_original']>=3 and defended_score['wer']<=.10
                        disrupt_identity=converted_score['asv']!=speaker
                        disrupt=disrupt_identity or converted_score['mos_original']<3 or converted_score['wer']>.10
                        protected_wave,_=sf.read(protected,dtype='float32');original_wave,_=sf.read(clean_path,dtype='float32')
                        delta=protected_wave-original_wave
                        crop_delta=delta[interval[0]:interval[1]]
                        excess=np.maximum(perturbation_psd(crop_delta,pp['n_fft'],pp['hop_length'],pp['win_length'],psd_max).T-theta,0)
                        row={'utterance':utterance,'content':content,'profile':name,'checkpoint':checkpoint,
                             'defended':defended_score,'converted':converted_score,'control':control,
                             'clean_identity_ok':identity_clean,'clean_quality_ok':clean_quality,
                             'protected_quality_ok':quality,'strict_joint':bool(identity_clean and quality and disrupt_identity),
                             'paper_structure_identity_screen':bool(identity_clean and quality and disrupt),
                             'paper_structure_full_screen':bool(identity_clean and clean_quality and quality and disrupt),
                             'mask_exceed_fraction':float(np.mean(excess>0)),'mask_excess_mean_db':float(excess.mean()),
                             'snr_db':float(10*np.log10(np.mean(original_wave**2)/np.mean(delta**2))),
                             'protocol_sha256':file_sha256(args.protocol)}
                        write_json(record,row);rows.append(row)
                        print(json.dumps({key:row[key] for key in ('utterance','content','profile','checkpoint','strict_joint','paper_structure_full_screen')}),flush=True)
                        write_json(out/(args.split+'-results.json'),{'rows':rows})
        state.update(state='complete',rows=len(rows))
    except BaseException as error:
        state.update(state='interrupted_or_failed',error=str(error));raise
    finally:
        state.update(elapsed_s=round(time.monotonic()-started,2),updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path,state)


if __name__=='__main__':main()
