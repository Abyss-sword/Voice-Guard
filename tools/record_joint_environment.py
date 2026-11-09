"""Record the inference environment after experiments, without changing settings."""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import write_json


def main():
    packages = {}
    for name in ('torch', 'torchaudio', 'numpy', 'scipy', 'librosa', 'soundfile',
                 'speechbrain', 'Resemblyzer', 'openai-whisper', 'h5py'):
        packages[name] = importlib.metadata.version(name)
    record = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(), 'packages': packages,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip(),
        'note': 'Read-only post-run package/hardware snapshot from a separate process, not a timestamped pre-run guarantee. Backend flags in a new process do not establish flags in the experiment process, so they are omitted. Fixed utterance seeds do not guarantee bitwise reproducibility of all CUDA kernels.'}
    for name in ('2026-11-09-joint', '2026-11-09-joint-anchored'):
        path = Path('docs/evidence') / name / 'runtime-environment.json'
        if path.exists(): raise ValueError('Environment snapshot already exists')
        write_json(path, record)
    print(record)


if __name__ == '__main__': main()
