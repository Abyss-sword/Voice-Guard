"""Download the official CTC checkpoint and verify its frozen study digest."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-content-robust'))
    args = parser.parse_args()
    protocol = json.loads((args.evidence / 'protocol.json').read_text(encoding='utf-8'))
    name = 'checkpoints/ctc/wav2vec2_fairseq_base_ls960_asr_ls960.pth'
    expected = protocol['assets'][name]
    target = Path(name)
    if target.exists():
        if file_sha256(target) != expected:
            raise ValueError('Existing checkpoint differs; preserving it rather than overwriting')
        print('Verified existing CTC checkpoint:', expected); return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.pth.part')
    if temporary.exists(): raise ValueError('Partial download exists; inspect it before retrying')
    from torch.hub import download_url_to_file
    download_url_to_file('https://download.pytorch.org/torchaudio/models/' + target.name,
        str(temporary), progress=True)
    if file_sha256(temporary) != expected:
        raise ValueError('Download digest differs; partial file preserved for inspection')
    temporary.rename(target)
    print('Verified official CTC checkpoint:', expected)


if __name__ == '__main__': main()
