"""Run the OFFICIAL AdaIN-VC inference end-to-end (with tensorflow stubbed out)."""
import sys
import types

# stub tensorflow (official utils.py imports it but get_spectrograms does not use it)
tf = types.ModuleType("tensorflow")
tf.cast = lambda x, **k: x
tf.minimum = lambda a, b: min(a, b)
sys.modules["tensorflow"] = tf

tbx = types.ModuleType("tensorboardX")


class _StubWriter:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


tbx.SummaryWriter = _StubWriter
sys.modules["tensorboardX"] = tbx

ed = types.ModuleType("editdistance")
ed.eval = lambda a, b: 0
sys.modules["editdistance"] = ed

# librosa 1.0 removed positional args from filters.mel; shim for the official code
import librosa  # noqa: E402
_orig_mel = librosa.filters.mel


def _mel_shim(*a, **k):
    if a:
        return _orig_mel(sr=a[0], n_fft=a[1], n_mels=a[2])
    return _orig_mel(**k)


librosa.filters.mel = _mel_shim

# librosa 1.0 made stft/istft keyword-only; shim for the official code
_orig_istft = librosa.istft


def _istft_shim(y, *a, **k):
    if len(a) >= 1:
        k.setdefault("hop_length", a[0])
    if len(a) >= 2:
        k.setdefault("win_length", a[1])
    return _orig_istft(y, **k)


librosa.istft = _istft_shim

_orig_stft = librosa.stft


def _stft_shim(y, *a, **k):
    if len(a) >= 1:
        k.setdefault("n_fft", a[0])
    if len(a) >= 2:
        k.setdefault("hop_length", a[1])
    return _orig_stft(y, **k)


librosa.stft = _stft_shim

sys.path.insert(0, r"D:\Voice Guard\third_party\ada_in_vc")
sys.path.insert(0, r"D:\Voice Guard\third_party\ada_in_vc\preprocess")

from preprocess.tacotron.utils import get_spectrograms, melspectrogram2wav  # noqa: E402
import pickle  # noqa: E402
import torch  # noqa: E402
from model import AE  # noqa: E402

cfg_path = r"D:\Voice Guard\third_party\ada_in_vc\config.yaml"
ckpt = r"D:\Voice Guard\checkpoints\adain-vc-official\vctk_model.ckpt"
attr_p = r"D:\Voice Guard\checkpoints\adain-vc-official\attr.pkl"
src = r"D:\Voice Guard\data\VCTK22\p226\p226_002.wav"
tgt = r"D:\Voice Guard\outputs\tmp_long_tgt.wav"
out = r"D:\Voice Guard\outputs\conv_official.wav"

import yaml
config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
model = AE(config).cuda().eval()
model.load_state_dict(torch.load(ckpt, map_location="cuda"))
attr = pickle.load(open(attr_p, "rb"))

src_mel, _ = get_spectrograms(src)
tar_mel, _ = get_spectrograms(tgt)
src_mel = torch.from_numpy((src_mel - attr["mean"]) / attr["std"]).cuda()
tar_mel = torch.from_numpy((tar_mel - attr["mean"]) / attr["std"]).cuda()
# official utt_make_frames (frame_size=1)
def frames(x):
    return x.view(1, x.size(0), x.size(1)).transpose(1, 2)
with torch.no_grad():
    dec = model.inference(frames(src_mel), frames(tar_mel))
dec = dec.transpose(1, 2).squeeze(0).cpu().numpy()
dec = dec * attr["std"] + attr["mean"]
wav = melspectrogram2wav(dec)
import soundfile as sf
sf.write(out, wav, 24000)
print("official convert saved:", out, len(wav) / 24000, "s")
