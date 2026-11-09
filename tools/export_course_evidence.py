"""Export cloud experiment evidence with optional WAV artifacts and a SHA256 receipt."""
from pathlib import Path
import argparse
import tarfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--audio',action='store_true')
    args=parser.parse_args()
    roots=[Path('outputs/cloud'),Path('outputs/course/2026-11-09'),Path('outputs/quality')]
    suffixes={'.json','.csv','.md','.log','.txt','.service'}|({'.wav','.png'} if args.audio else set())
    files=sorted(p for root in roots for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in suffixes)
    files += sorted(Path('outputs').glob('registry*.manifest.json'))
    sources=sorted(p for folder in ('vguard','scripts','deploy','platform/service','frontend','tools')
                   for p in Path(folder).rglob('*') if p.is_file() and p.suffix in {'.py','.sh','.service'})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.out,'w:gz') as archive:
        for p in files:archive.add(p,arcname=p.as_posix(),recursive=False)
        for p in sources:archive.add(p,arcname=f'executed-source/{p.as_posix()}',recursive=False)
    receipt={'archive':args.out.name,'bytes':args.out.stat().st_size,'sha256':file_sha256(args.out),
             'include_audio':args.audio,'files':len(files),'source_snapshot_files':len(sources),
             'source_snapshot_note':'Current cloud source bytes; stage manifests identify files actually used by each run'}
    write_json(args.out.with_suffix(args.out.suffix+'.receipt.json'),receipt)
    print(receipt)


if __name__=='__main__':main()
