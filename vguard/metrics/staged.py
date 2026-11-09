"""Cache paired independent evaluations; never exposed to waveform optimization."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .mos_original import OriginalMOSNet
from .wer import load_whisper, wer, cer, _edit_distance
from ..asv import load_ecapa, embed_wav
from ..run_state import file_sha256, write_json


class StagedEvaluator:
    def __init__(self, cache_path, device='cuda'):
        from resemblyzer import VoiceEncoder, preprocess_wav
        from whisper.normalizers import EnglishTextNormalizer
        self.device = device
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.mos = OriginalMOSNet(device=device)
        self.ecapa = load_ecapa(device=device)
        self.ge2e = VoiceEncoder(device=device, verbose=False)
        self.preprocess = preprocess_wav
        self.english = EnglishTextNormalizer()
        self.asr = {name: load_whisper(name, device) for name in ('base', 'small.en')}
        self.registries = {}
        for label, path in [('ecapa', 'outputs/registry.npz'), ('ge2e', 'outputs/paper_quality/ge2e-registry.npz')]:
            with np.load(path) as package:
                self.registries[label] = (list(package['names']), np.asarray(package['embeddings']))

    @torch.inference_mode()
    def score(self, path, reference):
        digest = file_sha256(path)
        key = digest + ':' + reference
        if key in self.cache: return self.cache[key]
        wave, sr = sf.read(path, dtype='float32')
        embeddings = {'ecapa': embed_wav(self.ecapa, wave, sr),
                      'ge2e': self.ge2e.embed_utterance(self.preprocess(wave, source_sr=sr))}
        result = {'sha256': digest, 'reference': reference, 'mos_original': self.mos.score(wave, sr)}
        for name, embedding in embeddings.items():
            names, registry = self.registries[name]
            sims = registry @ embedding / (np.linalg.norm(registry, axis=1) * np.linalg.norm(embedding) + 1e-9)
            result['asv_' + name] = str(names[int(sims.argmax())])
            result['similarities_' + name] = {str(n): float(v) for n, v in zip(names, sims)}
        for name, model in self.asr.items():
            label = 'small_en' if name == 'small.en' else name
            hypothesis = model.transcribe(str(path), language='en', temperature=0, fp16=False)['text'].strip()
            result['hypothesis_' + label] = hypothesis
            result['wer_' + label] = wer(hypothesis, reference)
            result['cer_' + label] = cer(hypothesis, reference)
            normalized_ref = self.english(reference).split()
            normalized_hyp = self.english(hypothesis).split()
            result['wer_english_' + label] = _edit_distance(normalized_hyp, normalized_ref) / len(normalized_ref) if normalized_ref else float('inf')
        self.cache[key] = result
        write_json(self.cache_path, self.cache)
        return result


def gates(utterance, original, clean_vc, defended, protected_vc, *, asv='ge2e', asr='small_en'):
    speaker = utterance.split('_')[0]
    identity = clean_vc['asv_' + asv] == speaker
    clean_quality = clean_vc['mos_original'] >= 3 and clean_vc['wer_' + asr] <= .10
    quality = defended['asv_' + asv] == speaker and defended['mos_original'] >= 3 and defended['wer_' + asr] <= .10
    changed = protected_vc['asv_' + asv] != speaker
    disruption = changed or protected_vc['mos_original'] < 3 or protected_vc['wer_' + asr] > .10
    return {'clean_identity_ok': bool(identity), 'clean_full_quality_ok': bool(identity and clean_quality),
            'protected_quality_ok': bool(quality), 'strict_joint': bool(identity and quality and changed),
            'paper_identity_screen': bool(identity and quality and disruption),
            'paper_full_screen': bool(identity and clean_quality and quality and disruption)}
