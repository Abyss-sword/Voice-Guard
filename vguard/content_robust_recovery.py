"""Experimental CTC-preserving, time-frequency allocated reference defense.

This implements a project extension, not a verified implementation of Kalypta.
Whisper, GE2E and MOSNet never enter the optimizer or its acceptance checks.
"""
from __future__ import annotations

import math
import re
import time
import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from torch import nn
from torch.nn import functional as F

from .joint_recovery import spectral_consistency
from .masking import MaskingProjection
from .recovery import constrain_delta
from .staged_recovery import NativeFeatures


class ContentCTC(nn.Module):
    """Frozen wav2vec2 ASR; gradients flow only to the input waveform."""
    def __init__(self, model=None, labels=None):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model(dl_kwargs={'model_dir': 'checkpoints/ctc'}) if model is None else model
        self.labels = tuple(bundle.get_labels() if labels is None else labels)
        if self.labels[0] != '-':
            raise ValueError('Expected blank at index zero')
        self.model.eval().requires_grad_(False)

    def target(self, text, device):
        # The LibriSpeech checkpoint predicts uppercase letters, apostrophe,
        # and |. This training normalization never changes evaluation WER.
        normalized = re.sub(r"[^A-Z' ]", ' ', text.upper())
        normalized = '|'.join(normalized.split())
        if not normalized:
            raise ValueError('Empty CTC training transcript')
        lookup = {value: index for index, value in enumerate(self.labels)}
        return torch.tensor([lookup[c] for c in normalized], device=device, dtype=torch.long)

    def loss(self, wave, sample_rate, text):
        if wave.ndim != 1 or not torch.isfinite(wave).all():
            raise ValueError('Finite mono waveform required')
        value = AF.resample(wave, sample_rate, 16000) if sample_rate != 16000 else wave
        emissions, _ = self.model(value.unsqueeze(0))
        targets = self.target(text, wave.device)
        if emissions.shape[1] < len(targets):
            raise ValueError('CTC transcript longer than emission sequence')
        result = F.ctc_loss(emissions.log_softmax(-1).transpose(0, 1), targets,
            torch.tensor([emissions.shape[1]], device=wave.device),
            torch.tensor([len(targets)], device=wave.device), blank=0,
            reduction='mean', zero_infinity=False)
        if not torch.isfinite(result):
            raise ValueError('Nonfinite CTC objective')
        return result


def first_order_rir(sample_rate, room, source, microphone, reflection=.25):
    """Deterministic first-order image sources; not a full room simulator.

    Remove direct-path delay and normalize the direct tap. Fractional echo
    delays are linearly interpolated. No diffuse tail or measured room claim.
    """
    room, source, microphone = [np.asarray(x, dtype=float) for x in (room, source, microphone)]
    if any(x.shape != (3,) for x in (room, source, microphone)) or not 0 <= reflection < 1:
        raise ValueError('Invalid image-source geometry')
    if np.any(source <= 0) or np.any(source >= room) or np.any(microphone <= 0) or np.any(microphone >= room):
        raise ValueError('Source and microphone must be inside the room')
    direct = float(np.linalg.norm(source - microphone))
    if direct < .05:
        raise ValueError('Distinct source and microphone required')
    paths = []
    for axis in range(3):
        for wall in (0., room[axis]):
            image = source.copy(); image[axis] = 2 * wall - source[axis]
            distance = float(np.linalg.norm(image - microphone))
            paths.append(((distance - direct) / 343. * sample_rate, reflection * direct / distance))
    taps = np.zeros(int(math.ceil(max(delay for delay, _ in paths))) + 2, dtype=np.float32)
    taps[0] = 1.
    for delay, gain in paths:
        position = int(math.floor(delay)); fraction = delay - position
        taps[position] += gain * (1 - fraction); taps[position + 1] += gain * fraction
    return torch.from_numpy(taps)


class ReferenceChannels(nn.Module):
    """Paired differentiable reference preprocessing; identity always included."""
    names = ('identity', 'small_room', 'larger_room', 'microphone', 'resample_8k')

    def __init__(self, sample_rate):
        super().__init__(); self.sample_rate = sample_rate
        self.register_buffer('small_room', first_order_rir(sample_rate,
            (4., 3.5, 2.7), (1., 1.2, 1.4), (2.5, 2., 1.2), .20))
        self.register_buffer('larger_room', first_order_rir(sample_rate,
            (7., 5., 3.), (1.2, 1.1, 1.5), (4.5, 3., 1.3), .32))
        # A fixed smooth frequency response with no room interpretation.
        self.register_buffer('microphone', torch.tensor([.80, .15, .05]))

    def forward(self, wave, index):
        if index == 0:
            return wave
        if index == 4:
            return AF.resample(AF.resample(wave, self.sample_rate, 8000), 8000,
                self.sample_rate)[:len(wave)]
        if index not in (1, 2, 3):
            raise ValueError('Unknown reference channel')
        taps = getattr(self, self.names[index])
        return AF.fftconvolve(wave, taps, mode='full')[:len(wave)]


def normalized_gradient(value):
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite input gradient')
    return value / value.square().mean().sqrt().clamp_min(1e-12)


def allocated_gradient(defense, content, n_fft=512, hop_length=128):
    """Allocate updates by relative identity/content gradient sensitivity.

    Gradient magnitudes are normalized separately; therefore their numerical
    loss scales do not alone decide the gate. The smoothed gate is in [.1,1].
    It is a local heuristic, not an audibility or transcript guarantee.
    """
    if defense.shape != content.shape or defense.ndim != 1:
        raise ValueError('Matching mono gradients required')
    gd, gc = normalized_gradient(defense), normalized_gradient(content)
    window = torch.hann_window(n_fft, device=gd.device, dtype=gd.dtype)
    def stft(x):
        return torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
    ds, cs = stft(gd), stft(gc)
    da, ca = ds.abs(), cs.abs()
    da = da / da.mean().clamp_min(1e-12)
    ca = ca / ca.mean().clamp_min(1e-12)
    gate = da / (da + ca + 1e-8)
    gate = F.avg_pool2d(gate[None, None], (5, 3), stride=1, padding=(2, 1),
        count_include_pad=False)[0, 0].clamp(.1, 1.)
    output = torch.istft(ds * gate, n_fft, hop_length, window=window, length=len(gd))
    return output, gate


def remove_content_conflict(defense, content):
    """Remove a first-order direction that increases CTC; no nonlinear promise."""
    dot = (defense * content).sum()
    norm = content.square().sum().clamp_min(1e-20)
    return defense - torch.minimum(dot, dot.new_zeros(())) / norm * content


def passes_proxy_caps(values, caps):
    if not all(math.isfinite(values[k]) for k in ('ctc', 'own_similarity', 'spectrum', 'native_defense')):
        return False
    return (values['ctc'] <= caps['ctc_max']
        and values['own_similarity'] >= caps['own_similarity_min']
        and values['spectrum'] <= caps['spectrum_max']
        and values['native_defense'] <= caps['native_defense_max'])


@torch.no_grad()
def check_export(model, original, exported, full_original, interval, preprocess, attr,
                 target_embeddings, ctc, transcript, surrogate):
    """Check the actual PCM16 export before any independent evaluator runs."""
    device = next(model.parameters()).device
    wave, changed, full = [torch.as_tensor(x, dtype=torch.float32, device=device)
        for x in (original, exported, full_original)]
    begin, end = map(int, interval)
    restored = torch.cat((full[:begin], changed, full[end:]))
    features = NativeFeatures(preprocess, attr).to(device)
    target = torch.as_tensor(target_embeddings, dtype=torch.float32, device=device).mean(0, keepdim=True)
    own, altered = model.speaker_encoder(features(wave)), model.speaker_encoder(features(changed))
    scale = F.mse_loss(own, target).clamp_min(1e-5)
    return {'ctc': float(ctc.loss(restored, preprocess['sample_rate'], transcript)),
        'own_similarity': float(F.cosine_similarity(surrogate(restored), surrogate(full)).mean()),
        'spectrum': float(spectral_consistency(wave, changed)),
        'native_defense': float((F.mse_loss(altered, target) - .1 * F.mse_loss(altered, own)) / scale)}


def refine_content(model, original, initial, theta, psd_max, attr, preprocess,
                   target_embeddings, ctc, transcript, full_original, interval,
                   surrogate, *, eot=True, steps=160, learning_rate=.00003,
                   snr_db=20., max_seconds=420., callback=None):
    """Refine one baseline without consulting an independent final evaluator.

    Proposals are accepted every eight updates only if the training CTC,
    original ECAPA similarity, spectrum and native clean-channel defense meet
    fixed baseline-relative caps and the mean training defense does not rise.
    A rejected block restores its previous accepted waveform. All choices and
    fallbacks are part of this optimizer, not final-metric checkpoint selection.
    """
    if steps < 1 or learning_rate <= 0 or max_seconds <= 0:
        raise ValueError('Invalid refinement budget')
    started = time.monotonic()
    device = next(model.parameters()).device
    wave, start, full = [torch.as_tensor(x, device=device, dtype=torch.float32)
        for x in (original, initial, full_original)]
    begin, end = map(int, interval)
    if start.shape != wave.shape or len(wave) != end - begin or not 0 <= begin < end <= len(full):
        raise ValueError('Protected crop and full context disagree')
    features = NativeFeatures(preprocess, attr).to(device)
    channels = ReferenceChannels(preprocess['sample_rate']).to(device)
    target = torch.as_tensor(target_embeddings, device=device, dtype=torch.float32).mean(0, keepdim=True)
    projector = MaskingProjection(theta, psd_max, **{k: preprocess[k]
        for k in ('n_fft', 'hop_length', 'win_length')}, margin_db=0.).to(device)
    indices = list(range(len(channels.names))) if eot else [0]

    def restore(crop):
        return torch.cat((full[:begin], crop, full[end:]))

    with torch.no_grad():
        own = [model.speaker_encoder(features(channels(wave, i))).detach() for i in indices]
        scales = [F.mse_loss(value, target).clamp_min(1e-5) for value in own]
        original_identity = surrogate(restore(wave)).detach()

    def defense_loss(crop, index):
        identity = model.speaker_encoder(features(channels(crop, indices[index])))
        return (F.mse_loss(identity, target) - .1 * F.mse_loss(identity, own[index])) / scales[index]

    def checks(crop):
        restored = restore(crop)
        return {'ctc': float(ctc.loss(restored, preprocess['sample_rate'], transcript)),
            'own_similarity': float(F.cosine_similarity(surrogate(restored), original_identity).mean()),
            'spectrum': float(spectral_consistency(wave, crop)),
            'native_defense': float(defense_loss(crop, 0)),
            'training_defense_mean': float(torch.stack([defense_loss(crop, i)
                for i in range(len(indices))]).mean())}

    with torch.no_grad():
        baseline = checks(start)
        caps = {'ctc_max': baseline['ctc'] + .01,
            'own_similarity_min': baseline['own_similarity'] - .005,
            'spectrum_max': baseline['spectrum'] + .01,
            'native_defense_max': baseline['native_defense'] + .005}
        delta = start - wave
        for _ in range(30):
            delta = constrain_delta(delta, wave, projector, snr_db, .1)
        accepted = (start - wave).clone()
        accepted_mean = baseline['training_defense_mean']
    delta = delta.detach().requires_grad_(True)
    trace = []; accepted_blocks = rejected_blocks = conflicts = 0
    rate = learning_rate
    for step in range(1, steps + 1):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('Content refinement per-utterance budget reached')
        crop = wave + delta
        defense = defense_loss(crop, (step - 1) % len(indices))
        content = ctc.loss(restore(crop), preprocess['sample_rate'], transcript)
        gd, = torch.autograd.grad(defense, delta)
        gc, = torch.autograd.grad(content, delta)
        if not torch.isfinite(gd).all() or not torch.isfinite(gc).all():
            raise ValueError('Nonfinite content/defense gradient')
        with torch.no_grad():
            direction, gate = allocated_gradient(gd, gc)
            conflicts += int(float((direction * gc).sum()) < 0)
            direction = remove_content_conflict(direction, gc)
            if float(content) > caps['ctc_max']:
                direction = direction + .5 * normalized_gradient(gc)
            direction = normalized_gradient(direction)
            delta.copy_(constrain_delta(delta - rate * direction, wave, projector, snr_db, .1))
            if step % 8 == 0 or step == steps:
                for _ in range(30):
                    delta.copy_(constrain_delta(delta, wave, projector, snr_db, .1))
                values = checks(wave + delta)
                allowed = (passes_proxy_caps(values, caps)
                    and values['training_defense_mean'] <= accepted_mean + .0001)
                if allowed:
                    accepted.copy_(delta); accepted_mean = values['training_defense_mean']; accepted_blocks += 1
                else:
                    delta.copy_(accepted); rejected_blocks += 1; rate = max(rate * .7, learning_rate / 16)
                record = {'step': step, 'accepted': allowed, 'checks': values,
                    'gate_mean': float(gate.mean()), 'learning_rate_next': rate,
                    'elapsed_s': round(time.monotonic() - started, 2)}
                trace.append(record); print(record, flush=True)
    result = (wave + accepted).detach().cpu().numpy()
    if callback:
        callback(result)
    return result, {'baseline_proxy_checks': baseline, 'proxy_caps': caps,
        'accepted_blocks': accepted_blocks, 'rejected_blocks': rejected_blocks,
        'content_gradient_conflicts': conflicts, 'trace': trace,
        'channels': [channels.names[i] for i in indices],
        'ctc_surrogate': 'torchaudio WAV2VEC2_ASR_BASE_960H; frozen, input gradients only',
        'defense_surrogate': 'AdaIN native reference encoder; no Griffin-Lim gradient',
        'original_identity_surrogate': 'ECAPA; participates in proxy acceptance checks',
        'independent_evaluators': ['GE2E', 'MOSNet', 'Whisper base', 'Whisper small.en'],
        'acceptance_note': 'Training-proxy constraints only; no independent-quality guarantee',
        'elapsed_s': round(time.monotonic() - started, 2)}
