"""Bounded course completion campaign; completed utterances resume in their own scripts.

Timeout stops this workload, not ECS billing. Final data/report status is explicit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256, write_json

SPEAKERS="p225,p226,p227,p228,p229"
OUT="outputs/course/2026-11-09"


def saved_quality_digest(command):
    """Keep completed measurements intact when resuming the campaign."""
    if not any(Path(arg).name in ('compute_mos.py','compute_wer.py') for arg in command):return None
    try:
        directory=Path(command[command.index('--exp-dir')+1])
        path=Path(command[command.index('--out')+1])
        with (directory/'results.csv').open(encoding='utf-8',newline='') as stream:
            expected={row['utterance'] for row in csv.DictReader(stream)}
        with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        prefix='mos' if any(Path(arg).name=='compute_mos.py' for arg in command) else 'wer'
        if len(expected)!=5 or len(rows)!=5 or {row['utterance'] for row in rows}!=expected:return None
        if prefix=='wer' and any(row.get('reference_mode')!='transcript' for row in rows):return None
        if not all(math.isfinite(float(row[f'{prefix}_{key}'])) for row in rows for key in
                   ('original_trimmed','defended','conv_from_clean','conv_from_defended')):return None
        return file_sha256(path)
    except (OSError,KeyError,ValueError,TypeError):return None


def verify_frozen_inputs(protocol):
    for item in protocol['new_utterances']:
        utterance=item['utterance']
        base=Path('data/VCTK22')/utterance.split('_')[0]/utterance
        for suffix,key in (('.wav','wav_sha256'),('.txt','transcript_sha256')):
            if file_sha256(base.with_suffix(suffix)) != item[key]:
                raise ValueError(f'Frozen input changed: {base}{suffix}')


def verify_completion(protocol):
    campaign=json.loads(Path('outputs/cloud/campaign-summary.json').read_text())
    if not campaign['complete']:
        raise ValueError('Original campaign is incomplete; inspect group summaries')
    expected={item['utterance'] for item in protocol['new_utterances']}
    for name in ('baseline','refined'):
        summary=json.loads((Path(OUT)/f'quality_new_{name}/summary.json').read_text())
        keys={(row['utterance'],row['scale']) for row in summary['rows']}
        if keys != {(utterance,scale) for utterance in expected for scale in (0,1)} or len(summary['rows'])!=20:
            raise ValueError(f'New utterance evaluation is incomplete: {name}')
    for prefix,multiplier in (('channels',8),('contents',6)):
        for name,count in (('baseline',5),('refined_dev',2),('refined_validation',3)):
            summary=json.loads((Path(OUT)/f'{prefix}_{name}/summary.json').read_text())
            if not summary.get('complete') or len(summary['rows']) != count*multiplier:
                raise ValueError(f'Audit incomplete: {prefix}_{name}')


def plan():
    py=sys.executable
    commands=[("preflight",[py,"tools/preflight.py","--scope","experiment","--out",f"{OUT}/preflight.json"]),
              ("resume_main",[py,"scripts/reproduce.py","main","--max-hours","2"]),
              ("resume_eps",[py,"scripts/reproduce.py","eps","--max-hours","2"])]
    for group in [f"main_{m}" for m in ("vg","vg-nop","emb","fb","e2e")]+[f"eps_{e}" for e in (".02",".05",".08",".10")]:
        directory=f"outputs/cloud/{group}"
        commands += [(f"mos_{group}",[py,"scripts/compute_mos.py","--exp-dir",directory,"--out",f"{directory}/mos.csv","--device","cuda"]),
                     (f"wer_{group}",[py,"scripts/compute_wer.py","--exp-dir",directory,"--vctk","data/VCTK22","--source-speaker","p226","--source-utt","2","--out",f"{directory}/wer.csv","--device","cuda"])]
    commands.append(("summarize_original_campaign",[py,"tools/build_cloud_report.py"]))
    # An independent utterance set; all settings fixed before any new quality evaluation.
    baseline=f"{OUT}/new_utterances_vg"
    refined=f"{OUT}/new_utterances_projected"
    commands += [("new_utterances_vg",[py,"scripts/run_experiment.py","--model-dir","checkpoints/adain-vc-official",
                    "--vctk","data/VCTK22","--speakers",SPEAKERS,"--per-speaker","2","--utt-offset","40",
                    "--max-dur","3.5","--source-speaker","p226","--source-utt","2","--registry","outputs/registry.npz",
                    "--attack-registry","outputs/registry_vc.npz","--eps",".05","--method","vg","--iters1","3000",
                    "--iters2","1500","--seed","2026","--device","cuda","--no-progress","--out-dir",baseline]),
                 ("new_utterances_refinement",[py,"scripts/refine_masked.py","--exp-dir",baseline,"--out-dir",refined,
                    "--speakers",SPEAKERS,"--steps","1000","--learning-rate",".0003","--margin-db","3"])]
    for name,directory in (("baseline",baseline),("refined",refined)):
        commands.append((f"quality_new_{name}",[py,"scripts/quality_sweep.py","--exp-dir",directory,
                         "--out-dir",f"{OUT}/quality_new_{name}","--speakers",SPEAKERS,"--scales","0,1","--wav-subtype","PCM_16"]))
    # Compare the original method and the frozen improvement on matched codecs and sources.
    for name,directory,speakers in (("baseline","outputs/cloud/main_vg",SPEAKERS),
                                   ("refined_dev","outputs/quality/refine_dev","p225,p226"),
                                   ("refined_validation","outputs/quality/refine_validation","p227,p228,p229")):
        commands.append((f"channels_{name}",[py,"scripts/audit_channels.py","--exp-dir",directory,
                        "--out-dir",f"{OUT}/channels_{name}","--speakers",speakers]))
        commands.append((f"contents_{name}",[py,"scripts/audit_channels.py","--exp-dir",directory,
                        "--out-dir",f"{OUT}/contents_{name}","--speakers",speakers,
                        "--channels","none,pcm16","--contents","p226_002,p225_002,p227_002"]))
    return commands


def execute(command,timeout,log):
    with log.open("ab") as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                 start_new_session=os.name!="nt")
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            if os.name=="nt": process.terminate()
            else: os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name=="nt": process.kill()
                else: os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hours",type=float,default=3)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    commands=plan()
    if args.dry_run:
        print(json.dumps(commands,ensure_ascii=False,indent=2));return
    if not 0 < args.max_hours <= 6: parser.error("max-hours must be in (0,6]")
    os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1")
    os.chdir(ROOT)
    out=Path(OUT);out.mkdir(parents=True,exist_ok=True)
    protocol=Path("docs/evidence/2026-11-09/frozen-protocol.json")
    frozen=json.loads(protocol.read_text(encoding='utf-8'))
    verify_frozen_inputs(frozen)
    record={"started_utc":datetime.now(timezone.utc).isoformat(),"state":"running","max_hours":args.max_hours,
            "protocol_sha256":file_sha256(protocol),"commands":commands,"stages":[],
            "billing_note":"Workload time limit does not power off ECS"}
    state=out/f"campaign-{time.time_ns()}.json"
    started=time.monotonic()
    try:
        for name,command in commands:
            remaining=args.max_hours*3600-(time.monotonic()-started)
            if remaining<=0: raise subprocess.TimeoutExpired(command,args.max_hours*3600)
            print(name,flush=True)
            item={"name":name,"started_utc":datetime.now(timezone.utc).isoformat(),"log":str(out/f"{name}.log")}
            record["stages"].append(item)
            write_json(state,record)
            stage_started=time.monotonic()
            preserved=saved_quality_digest(command)
            if preserved:
                code=0
                item['preserved_quality_csv_sha256']=preserved
                item['quality_note']='Existing complete scores preserved; not recomputed by this invocation'
            else:
                code=execute(command,remaining,Path(item["log"]))
            item.update(returncode=code,elapsed_s=round(time.monotonic()-stage_started,2))
            write_json(state,record)
            if code: raise subprocess.CalledProcessError(code,command)
        verify_completion(frozen)
        record["state"]="complete"
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,KeyboardInterrupt,ValueError) as error:
        record["state"]="interrupted_or_failed"
        record["error"]=str(error)
        raise
    finally:
        record["elapsed_s"]=round(time.monotonic()-started,2)
        record["updated_utc"]=datetime.now(timezone.utc).isoformat()
        write_json(state,record)


if __name__=="__main__":main()
