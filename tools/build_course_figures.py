"""Generate report figures from measured WAV/JSON/CSV artifacts, without model scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.audio import load_audio
from vguard.masking import generate_th, perturbation_psd


def save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-dir",type=Path,default=ROOT/"outputs/course/remote/outputs/course/2026-11-09")
    parser.add_argument("--cloud-dir",type=Path,default=ROOT/"outputs/course/remote/outputs/cloud")
    parser.add_argument("--out",type=Path,default=ROOT/"docs/figures/2026-11-09")
    args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    comparison=json.loads((ROOT/"docs/evidence/2026-11-04/quality/comparison.json").read_text(encoding="utf-8"))
    rows=comparison["rows"]
    x=np.arange(len(rows));width=.36
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6))
    for ax,key,title,threshold in ((axes[0],"mos","Protected MOSNet prediction",3),
                                  (axes[1],"wer","Protected word error rate",10)):
        scale=100 if key=="wer" else 1
        ax.bar(x-width/2,[row[f"baseline_{key}"]*scale for row in rows],width,label="Original VG",color="#8493a5")
        ax.bar(x+width/2,[row[f"refined_{key}"]*scale for row in rows],width,label="Projected refinement",color="#217a78")
        ax.axhline(threshold,color="#b23a48",ls="--",lw=1,label="Engineering gate")
        ax.set_xticks(x,[row["utterance"].split('_')[0] for row in rows])
        ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.15)
        if key=="wer":ax.set_ylabel("Percent; macro average reported separately")
    axes[0].legend(fontsize=8,loc="lower left")
    fig.text(.5,-.03,"All five fixed utterances; p229 excluded only from defense denominator. Predictor scores are not human MOS.",ha="center",fontsize=8)
    fig.tight_layout();save(fig,out/"01_quality_comparison.png")

    audio=ROOT/"outputs/deploy/quality-evidence/outputs"
    baseline_dir=audio/"cloud/main_vg/p225_040"
    original,sr=sf.read(baseline_dir/"original_trimmed.wav",dtype="float32")
    base,_=sf.read(baseline_dir/"defended.wav",dtype="float32")
    refined,_=sf.read(audio/"quality/refine_dev_pcm16_eval/p225_040/scale_1/defended.wav",dtype="float32")
    fig,axes=plt.subplots(3,1,figsize=(10,5.4),sharex=True)
    time=np.arange(len(original))/sr
    for ax,wave,title,color in zip(axes,[original,base,refined],["Original trimmed speech","Original VG output","Projected refinement (PCM16)"],["#4d5968","#b23a48","#217a78"]):
        ax.plot(time,wave,lw=.45,color=color)
        ax.set_ylabel("Amplitude");ax.set_title(title,loc="left",fontsize=10)
        ax.set_ylim(-1,1)
    axes[-1].set_xlabel("Time (s)");fig.tight_layout();save(fig,out/"02_waveform_p225.png")

    cfg=dict(n_fft=2048,hop_length=300,win_length=1200)
    theta,psd_max=generate_th(original,sr,**cfg)
    frequencies=np.linspace(0,sr/2,theta.shape[1])
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6))
    axes[0].plot(frequencies/1000,theta.mean(axis=0),label="Mean masking threshold",color="#4d5968")
    excesses=[]
    for wave,label,color in ((base,"Original VG","#b23a48"),(refined,"Projected refinement","#217a78")):
        psd=perturbation_psd(wave-original,**cfg,psd_max=psd_max).T
        excess=np.maximum(psd-theta,0)
        excesses.append({"method":label,"mean_positive_excess_db":float(excess.mean()),
                         "exceed_fraction":float((excess>0).mean())})
        axes[0].plot(frequencies/1000,psd.mean(axis=0),label=label,color=color,alpha=.85)
        axes[1].plot(np.arange(len(excess))*cfg["hop_length"]/sr,excess.mean(axis=1),label=label,color=color)
    axes[0].set(xlabel="Frequency (kHz)",ylabel="Normalized PSD (dB)",title="Averages do not prove every bin is masked")
    axes[1].set(xlabel="Time (s)",ylabel="Mean positive excess (dB)",title="Measured per-frame excess over threshold")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();save(fig,out/"03_masking_p225.png")
    (out/"masking_p225.json").write_text(json.dumps(excesses,indent=2)+'\n',encoding="utf-8")

    import librosa
    full=load_audio(ROOT/"data/VCTK22/p227/p227_033.wav",24000)
    crop,interval=librosa.effects.trim(full,top_db=15)
    fig,ax=plt.subplots(figsize=(10,2.8))
    ax.plot(np.arange(len(full))/24000,full,lw=.55,color="#4d5968")
    ax.axvspan(0,interval[0]/24000,color="#b23a48",alpha=.18,label="Removed by top_db=15")
    ax.axvspan(interval[1]/24000,len(full)/24000,color="#b23a48",alpha=.18)
    ax.axvline(interval[0]/24000,color="#b23a48",ls="--",lw=.8)
    ax.set(xlabel="Time (s)",ylabel="Amplitude",title="p227_033: crop removes the quiet opening; measured WER 28.57% to 0 with context")
    ax.legend(fontsize=8);fig.tight_layout();save(fig,out/"04_crop_p227.png")

    summary=args.cloud_dir/"campaign-summary.json"
    if summary.exists():
        groups=json.loads(summary.read_text(encoding="utf-8"))["groups"]
        eps=[group for group in groups if group["name"].startswith("eps_")]
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        x=[float(group["name"].split('_')[1]) for group in eps]
        axes[0].plot(x,[100*group["screened_rate"] if group["screened_rate"] is not None else np.nan for group in eps],marker='o',color='#217a78')
        for value,group in zip(x,eps):
            if group['screened_rate'] is not None:
                axes[0].annotate(f"{group['screened_successes']}/{group['screened']}",(value,100*group['screened_rate']),xytext=(0,8),textcoords='offset points',ha='center')
        axes[0].set(xlabel="Epsilon",ylabel="Screened identity success (%)",ylim=(-5,110),title="Identity-only success; not joint quality")
        axes[1].plot(x,[group["mean_mos_defended"] if group["mean_mos_defended"] is not None else np.nan for group in eps],marker='o',color='#b23a48')
        axes[1].axhline(3,color='#4d5968',ls='--',lw=.8)
        axes[1].set(xlabel="Epsilon",ylabel="Protected MOSNet prediction",title="All five fixed utterances per epsilon")
        fig.tight_layout();save(fig,out/"05_epsilon_tradeoff.png")

    paths=[args.course_dir/f'quality_new_{name}/summary.json' for name in ('baseline','refined')]
    if all(path.exists() for path in paths):
        paired=[{row['utterance']:row for row in json.loads(path.read_text(encoding='utf-8'))['rows']
                 if row['scale']==1} for path in paths]
        if len(paired[0])==10 and set(paired[0])==set(paired[1]):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,metric,label,scale,gate in ((axes[0],'mos_defended','MOSNet prediction',1,3),
                                             (axes[1],'wer_defended','Word error rate (%)',100,10)):
                x=[paired[0][key][metric]*scale for key in sorted(paired[0])]
                y=[paired[1][key][metric]*scale for key in sorted(paired[0])]
                hi=max(max(x+y)*1.1,gate*1.1);lo=min(-.025*hi,min(x+y))
                ax.plot([lo,hi],[lo,hi],color='#8493a5',ls='--',lw=1,label='No change')
                from collections import Counter
                counts=Counter(zip(x,y))
                ax.scatter([point[0] for point in counts],[point[1] for point in counts],
                           color='#217a78',s=[48*count for count in counts.values()],alpha=.75,label='Paired utterance (n=10)')
                for (px,py),count in counts.items():
                    if count>1:ax.annotate(f'x{count}',(px,py),xytext=(6,7),textcoords='offset points',fontsize=8)
                ax.axhline(gate,color='#b23a48',ls=':',lw=1,label='Refined quality gate')
                ax.set(xlabel=f'Original VG: {label}',ylabel=f'Projected: {label}',xlim=(lo,hi),ylim=(lo,hi))
                ax.grid(alpha=.15);ax.legend(fontsize=8)
            fig.suptitle('Frozen new utterances: quality improvement does not imply identity defense',fontsize=11)
            fig.tight_layout();save(fig,out/'07_new_utterances.png')

    audits=[]
    for name in ("channels_baseline","channels_refined_dev","channels_refined_validation"):
        path=args.course_dir/name/"summary.json"
        if path.exists():audits.append((name,json.loads(path.read_text(encoding="utf-8"))))
    if len(audits)==3:
        fig,axes=plt.subplots(1,2,figsize=(11.5,3.8))
        labels=audits[0][1]["protocol"]["channels"]
        x=np.arange(len(labels))
        for method,color in (("baseline","#8493a5"),("refined","#217a78")):
            rows=[row for name,audit in audits if method in name for row in audit["rows"]]
            successes=[];quality=[]
            for channel in labels:
                subset=[row for row in rows if row['channel']==channel]
                eligible=[row for row in subset if row['clean_vc_identity_ok']]
                successes.append(100*sum(row['screened_quality_identity_ok'] for row in eligible)/len(eligible) if eligible else np.nan)
                quality.append(np.mean([row['mos_defended'] for row in subset]))
            axes[0].plot(x,successes,marker='o',label=method,color=color)
            axes[1].plot(x,quality,marker='o',label=method,color=color)
        for ax in axes:
            ax.set_xticks(x,labels,rotation=35,ha='right');ax.legend(fontsize=8);ax.grid(alpha=.15)
        axes[0].set(ylabel='Screened joint pass (%)',ylim=(-5,110),title='Every channel has a matched clean-VC control')
        axes[1].set(ylabel='Protected MOSNet prediction',title='Laboratory codec roundtrips; five utterances')
        fig.tight_layout();save(fig,out/"06_channels.png")
    print('Figures:',out)


if __name__=="__main__":main()
