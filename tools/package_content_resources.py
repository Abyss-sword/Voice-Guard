"""Package the exact supplementary CTC checkpoint for offline reproduction."""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    evidence = Path('docs/evidence/2026-11-09-content-robust')
    protocol = json.loads((evidence / 'protocol.json').read_text(encoding='utf-8'))
    checkpoint = Path('checkpoints/ctc/wav2vec2_fairseq_base_ls960_asr_ls960.pth')
    expected = protocol['assets'][checkpoint.as_posix()]
    if file_sha256(checkpoint) != expected: raise ValueError('CTC checkpoint differs')
    output = Path('outputs/content_robust/VoiceGuard-content-ctc-resources.zip')
    if output.exists(): raise ValueError('Preserve existing CTC resource package')
    text = ('Supplementary resource for the frozen Voice Guard content-supervised study.\n'
        'Extract at the project root, preserving checkpoints/ctc/.\n'
        'Use with the existing complete model and quality-evaluator resource packages.\n'
        'Checkpoint: torchaudio WAV2VEC2_ASR_BASE_960H, English LibriSpeech 960h.\n'
        'Source: https://download.pytorch.org/torchaudio/models/' + checkpoint.name + '\n'
        'Tested torch/torchaudio: 2.5.1+cu124 on Ubuntu, Python 3.10.\n'
        'SHA256: ' + expected + '\n'
        'This is a training-content surrogate, not an independent final evaluator.\n'
        'It is not a Kalypta checkpoint or SDK.\n')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as package:
        package.write(checkpoint, checkpoint.as_posix())
        package.writestr('CONTENT-CTC-README.txt', text)
    receipt = {'archive': output.name, 'bytes': output.stat().st_size,
        'sha256': file_sha256(output), 'checkpoint_path': checkpoint.as_posix(),
        'checkpoint_sha256': expected, 'members': 2,
        'source': 'https://download.pytorch.org/torchaudio/models/' + checkpoint.name,
        'note': 'Supplement to existing model/evaluator packages; no private credentials or experiment scores.'}
    write_json(evidence / 'ctc-resource-receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
