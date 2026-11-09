"""Verify an exported archive, safely extract it, and retain compact review evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=ROOT/'outputs/course/remote')
    parser.add_argument('--evidence',type=Path,default=ROOT/'docs/evidence/2026-11-09/cloud')
    args=parser.parse_args()
    receipt=json.loads(args.receipt.read_text(encoding='utf-8'))
    digest=sha256(args.archive)
    if digest!=receipt['sha256'] or args.archive.stat().st_size!=receipt['bytes']:
        raise ValueError('Archive hash/size does not match receipt')
    args.destination.mkdir(parents=True,exist_ok=True)
    destination=args.destination.resolve()
    with tarfile.open(args.archive,'r:gz') as archive:
        for member in archive.getmembers():
            posix=PurePosixPath(member.name)
            target=(destination/member.name).resolve()
            if (posix.is_absolute() or '..' in posix.parts or ':' in member.name
                    or not target.is_relative_to(destination) or not member.isfile()):
                raise ValueError(f'Unsafe archive member: {member.name}')
        archive.extractall(destination,filter='data')
    artifact_checks={}
    for root in ('outputs/course/2026-11-09','outputs/quality'):
        for summary in (destination/root).rglob('summary.json'):
            obj=json.loads(summary.read_text(encoding='utf-8'))
            for row in obj.get('rows',[]):
                artifacts=row.get('artifacts_sha256',{})
                if artifacts:
                    selected=[(summary.parent/name,digest) for name,digest in artifacts.items()]
                elif 'audio_sha256' in row and 'converted_sha256' in row:
                    folder=summary.parent/row['utterance']/f"scale_{row['scale']:g}"
                    selected=[(folder/'defended.wav',row['audio_sha256']),
                              (folder/'converted.wav',row['converted_sha256'])]
                else:continue
                for path,expected in selected:
                    if not path.resolve().is_relative_to(destination):raise ValueError('Unsafe artifact path')
                    actual=artifact_checks.setdefault(str(path),sha256(path))
                    if actual!=expected:raise ValueError(f'Artifact differs from experiment record: {path}')
    copied=[]
    for folder in ('outputs/cloud','outputs/course/2026-11-09','outputs/quality'):
        for source in sorted((destination/folder).rglob('*')):
            if not source.is_file() or source.suffix not in {'.json','.csv','.md','.log','.txt'}:continue
            relative=source.relative_to(destination/'outputs')
            target=args.evidence/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            copied.append({'path':relative.as_posix(),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    record={**receipt,'local_archive_verified':True,'experiment_artifacts_verified':len(artifact_checks),
            'copied_text_files':len(copied),'text_sha256':copied}
    args.evidence.mkdir(parents=True,exist_ok=True)
    (args.evidence.parent/'final-archive.receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:record[key] for key in ('sha256','bytes','experiment_artifacts_verified','copied_text_files','local_archive_verified')},indent=2))


if __name__=='__main__':main()
