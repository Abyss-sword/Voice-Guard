"""Archive audited VQMIVC transfer evidence and its separate released resources."""
from datetime import datetime,timezone
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def bundle(path,entries,extra=None):
    receipt=Path(str(path)+'.receipt.json')
    if path.exists() or receipt.exists():raise ValueError('Preserve immutable package: '+str(path))
    with tarfile.open(path,'w:gz') as out:
        for name,source in sorted(entries.items()):out.add(source,arcname=name,recursive=False)
    record={'archive':path.name,'bytes':path.stat().st_size,'sha256':file_sha256(path),
        'files_sha256':{name:file_sha256(source) for name,source in sorted(entries.items())},
        'wav_files':sum(n.endswith('.wav') for n in entries),**(extra or {})}
    write_json(receipt,record)
    print(json.dumps({k:v for k,v in record.items() if k!='files_sha256'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path('docs/evidence/2026-11-09-vqmivc'))
    parser.add_argument('--archive',type=Path,default=Path('outputs/vqmivc/transfer-evidence.tar.gz'))
    parser.add_argument('--skip-resources',action='store_true')
    args=parser.parse_args()
    evidence=args.evidence;read=lambda p:json.loads(Path(p).read_text())
    p=read(evidence/'protocol.json');run=Path(p['output'])
    if not read(evidence/'analysis.json')['counts_verified'] or read(run/'state.json')['state']!='complete':
        raise ValueError('Complete audited transfer required')
    for name,digest in {**p['assets'],**p['code_sha256']}.items():
        if file_sha256(name)!=digest:raise ValueError('Frozen input changed: '+name)
    import torch
    optional={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=['outputs/vqmivc/vendor'])}
    write_json(evidence/'runtime-environment.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'optional_isolated_packages':optional,
        'setup':'python -m pip install --no-deps --target outputs/vqmivc/vendor pyworld==0.3.5 resampy==0.4.3',
        'launch_environment':{'PYTHONPATH':'outputs/vqmivc/vendor','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        'primary_environment_modified':False,'automatic_shutdown':False})
    assets={name:Path(name) for name in p['assets'] if name.startswith('checkpoints/vqmivc-official/')}
    assets['checkpoints/vqmivc-official/download-receipt.json']=Path('checkpoints/vqmivc-official/download-receipt.json')
    if not args.skip_resources:
        bundle(Path('outputs/vqmivc/vqmivc-official-resources.tar.gz'),assets,
            {'upstream_release':'https://github.com/Wendison/VQMIVC/releases/tag/v0.0',
             'note':'Locally recorded digests; no independent upstream digest established.'})
    entries={name:Path(name) for name in p['assets'] if not name.startswith('checkpoints/')}
    for directory in (run,evidence,Path('outputs/vqmivc/smoke'),Path('third_party/vqmivc'),
                      Path('outputs/vqmivc/clean-control-diagnosis')):
        for path in directory.rglob('*'):
            if path.is_file() and not path.is_symlink():entries[path.as_posix()]=path
    for path in Path('outputs/vqmivc').glob('*.log'):entries[path.as_posix()]=path
    for path in Path('outputs/vqmivc').glob('*attempt-state.json'):entries[path.as_posix()]=path
    for name in (*p['code_sha256'],'tools/audit_vqmivc_transfer.py','tools/archive_vqmivc_transfer.py',
                 'tools/diagnose_vqmivc_controls.py','tools/diagnose_vqmivc_resampling.py',
                 'tools/audit_vqmivc_transport.py'):
        entries['executed-source/'+name]=Path(name)
    bundle(args.archive,entries,
        {'limited_observed_sentence_transfer':True,'rows':read(evidence/'analysis.json')['rows']})


if __name__=='__main__':main()
