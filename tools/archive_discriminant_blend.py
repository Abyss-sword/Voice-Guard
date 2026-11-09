"""Archive the audited global-blend development and validation evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    evidence=Path('docs/evidence/2026-11-09-discriminant-blend');read=lambda p:json.loads(Path(p).read_text())
    p=read(evidence/'protocol.json');run=Path(p['output']);analysis=read(evidence/'analysis.json')
    if not analysis['counts_verified'] or read(run/'development-state.json')['state']!='complete' or read(run/'validation-state.json')['state']!='complete':
        raise ValueError('Complete audited blend required')
    for name,digest in {**p['assets'],**p['code_sha256']}.items():
        if file_sha256(name)!=digest:raise ValueError('Frozen input changed: '+name)
    archive=run.parent/'blend-evidence.tar.gz';receipt=Path(str(archive)+'.receipt.json')
    if archive.exists() or receipt.exists():raise ValueError('Preserve immutable blend archive')
    entries={name:Path(name) for name in p['assets'] if not name.startswith('checkpoints/')}
    for directory in (run,evidence):
        for path in directory.rglob('*'):
            if path.is_file() and not path.is_symlink():entries[path.as_posix()]=path
    for name in (*p['code_sha256'],'tools/audit_blend_results.py','tools/archive_discriminant_blend.py'):
        entries['executed-source/'+name]=Path(name)
    with tarfile.open(archive,'w:gz') as out:
        for name,path in sorted(entries.items()):out.add(path,arcname=name,recursive=False)
    record={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':file_sha256(archive),
        'files_sha256':{name:file_sha256(path) for name,path in sorted(entries.items())},
        'wav_files':sum(name.endswith('.wav') for name in entries),'rows_development':192,'rows_validation':160,
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'automatic_shutdown':False}
    write_json(receipt,record);write_json(evidence/'archive-verification.json',{k:v for k,v in record.items() if k!='files_sha256'}|{'all_files_verified':len(entries)})
    print(json.dumps({k:v for k,v in record.items() if k!='files_sha256'},indent=2))


if __name__=='__main__':main()
