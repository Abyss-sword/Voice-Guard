"""Experimental output-waveform defense with a frozen ECAPA surrogate.

GE2E, MOSNet and Whisper are held out of this optimizer. Griffin--Lim uses the
official forward iteration count, with a deliberately truncated phase gradient.
That gradient is a surrogate, not the exact derivative of all 100 iterations.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torchaudio.functional as AF
from torch import nn
from torch.nn import functional as F

from .masking import MaskingProjection
from .recovery import constrain_delta
from .staged_recovery import NativeFeatures
from .vc.adainvc import inv_mel_matrix


class WaveformReconstructor(nn.Module):
    """Official mel inversion / GL / de-emphasis, using a bounded gradient graph.

    The returned waveform is untrimmed. Silence trimming is handled by the
    surrogate encoder's fixed crop, rather than differentiated through.
    """
    def __init__(self, preprocess, attr, *, gradient_iterations=2):
        super().__init__()
        self.pp = dict(preprocess)
        if not 0 <= gradient_iterations <= preprocess['n_iter']:
            raise ValueError('Invalid Griffin-Lim gradient iteration count')
        self.gradient_iterations = gradient_iterations
        self.register_buffer('mean', torch.as_tensor(attr['mean'], dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(attr['std'], dtype=torch.float32))
        self.register_buffer('inverse', torch.as_tensor(inv_mel_matrix(
            preprocess['sample_rate'], preprocess['n_fft'], preprocess['n_mels'])))
        self.register_buffer('window', torch.hann_window(preprocess['win_length'], dtype=torch.float64))

    def magnitude(self, normalized_mel):
        if normalized_mel.ndim != 3 or normalized_mel.shape[0] != 1:
            raise ValueError('Expected one [1, mel, frames] spectrogram')
        mel = (normalized_mel[0].T * self.std + self.mean).T.clamp(0, 1)
        amplitude = torch.pow(10., (mel * self.pp['max_db'] - self.pp['max_db'] + self.pp['ref_db']) * .05)
        # The official NumPy inverse matrix is float64. Retaining that dtype
        # matters for phase convergence over 100 iterations.
        return self.inverse @ amplitude.to(self.inverse.dtype)

    def synthesis(self, spectrum):
        return torch.istft(spectrum, n_fft=self.pp['n_fft'], hop_length=self.pp['hop_length'],
            win_length=self.pp['win_length'], window=self.window, center=True)

    def phase(self, spectrum):
        wave = self.synthesis(spectrum)
        estimate = torch.stft(wave, n_fft=self.pp['n_fft'], hop_length=self.pp['hop_length'],
            win_length=self.pp['win_length'], window=self.window, center=True,
            pad_mode='constant', return_complex=True)
        return estimate / estimate.abs().clamp_min(1e-8)

    def forward(self, normalized_mel):
        magnitude = self.magnitude(normalized_mel)
        cold_steps = self.pp['n_iter'] - self.gradient_iterations
        # Every forward pass still performs the configured total GL iterations.
        with torch.no_grad():
            spectrum = magnitude.detach().to(torch.complex128)
            for _ in range(cold_steps):
                spectrum = magnitude.detach() * self.phase(spectrum)
            phase = spectrum / magnitude.detach().clamp_min(1e-8)
        spectrum = magnitude * phase.detach()
        for _ in range(self.gradient_iterations):
            spectrum = magnitude * self.phase(spectrum)
        wave = self.synthesis(spectrum)
        a = wave.new_tensor([1., -self.pp['preemph']])
        b = wave.new_tensor([1., 0.])
        return AF.lfilter(wave, a, b, clamp=False).to(normalized_mel.dtype)


class SurrogateSpeaker(nn.Module):
    """Differentiable ECAPA; evaluation wrappers elsewhere remain unchanged."""
    def __init__(self, classifier, sample_rate):
        super().__init__()
        self.classifier = classifier
        self.sample_rate = sample_rate
        self.classifier.eval().requires_grad_(False)

    def forward(self, wave):
        if self.sample_rate != 16000:
            wave = AF.resample(wave, self.sample_rate, 16000)
        return F.normalize(self.classifier.encode_batch(wave.unsqueeze(0)).reshape(1, -1), dim=-1)


def identity_margin(embedding, registry, own_index):
    """Positive means the source identity beats every other enrolled identity."""
    similarities = embedding @ F.normalize(registry, dim=-1).T
    own = similarities[..., own_index]
    others = torch.cat((similarities[..., :own_index], similarities[..., own_index + 1:]), dim=-1)
    return own - others.max(dim=-1).values


def spectral_consistency(original, protected):
    """Multi-resolution spectral preservation surrogate, not an MOS estimate."""
    losses = []
    for n_fft in (256, 512, 1024):
        window = torch.hann_window(n_fft, device=original.device)
        def magnitude(value):
            return torch.stft(value, n_fft=n_fft, hop_length=n_fft // 4,
                window=window, return_complex=True).abs()
        clean = magnitude(original).detach()
        changed = magnitude(protected)
        losses.append((changed - clean).abs().mean() / clean.mean().clamp_min(1e-5))
    return torch.stack(losses).mean()


def refine_joint(model, original, initial, content_mels, theta, psd_max, attr, preprocess,
                 surrogate, registry, own_index, *, steps=240, learning_rate=.00015,
                 snr_db=20., guarded=True, gradient_iterations=2, max_seconds=600.,
                 callback=None):
    """Refine a fixed baseline using reconstructed-waveform identity separation.

    Each step uses one fixed development content (cyclically). The reference
    waveform constraints are enforced after every update and at final export.
    No final evaluator, transcript, or validation content enters this function.
    """
    if steps < 1 or max_seconds <= 0 or not content_mels or learning_rate <= 0:
        raise ValueError('Invalid joint optimization settings')
    device = next(model.parameters()).device
    wave = torch.as_tensor(original, dtype=torch.float32, device=device)
    start = torch.as_tensor(initial, dtype=torch.float32, device=device)
    if start.shape != wave.shape or not torch.isfinite(start).all():
        raise ValueError('Initial protected waveform must match the original')
    registry = torch.as_tensor(registry, dtype=torch.float32, device=device)
    if registry.ndim != 2 or not 0 <= own_index < registry.shape[0] or registry.shape[0] < 2:
        raise ValueError('Invalid surrogate enrollment')
    features = NativeFeatures(preprocess, attr).to(device)
    reconstructor = WaveformReconstructor(preprocess, attr, gradient_iterations=gradient_iterations).to(device)
    projector = MaskingProjection(theta, psd_max, **{k: preprocess[k] for k in
        ('n_fft', 'hop_length', 'win_length')}, margin_db=0.).to(device)
    with torch.no_grad():
        clean_features = features(wave)
        clean_content = model.content_encoder(clean_features)[0].detach()
        content_scale = clean_content.square().mean().clamp_min(1e-5)
        own_embedding = surrogate(wave).detach()
        contents = [model.content_encoder(value.to(device))[0].detach() for value in content_mels]
        delta = start - wave
        for _ in range(30):
            delta = constrain_delta(delta, wave, projector, snr_db, .1)
    delta = delta.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    guard_weight = 2.
    trace = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('Joint refinement per-utterance time limit reached')
        protected = wave + delta
        mel = features(protected)
        embedding = model.speaker_encoder(mel)
        index = (step - 1) % len(contents)
        converted = reconstructor(model.decoder(contents[index], embedding))
        output_margin = identity_margin(surrogate(converted), registry, own_index).mean()
        # Target a small negative classification margin, rather than an arbitrary
        # distant speaker. Retain a gradient until the conversion crosses it.
        defense = F.relu(output_margin + .05)
        if guarded:
            protected_embedding = surrogate(protected)
            preservation = F.relu(.98 - F.cosine_similarity(protected_embedding, own_embedding)).mean()
            preservation = preservation + F.relu(.03 - identity_margin(protected_embedding, registry, own_index)).mean()
            content = F.mse_loss(model.content_encoder(mel)[0], clean_content) / content_scale
            spectrum = spectral_consistency(wave, protected)
            guard = preservation + .2 * F.relu(content - .02) + .1 * F.relu(spectrum - .08)
        else:
            guard = defense.new_zeros(())
            preservation = content = spectrum = guard
        loss = defense + guard_weight * guard
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise ValueError('Invalid joint waveform gradient')
        torch.nn.utils.clip_grad_norm_([delta], 1.)
        optimizer.step()
        with torch.no_grad():
            delta.copy_(constrain_delta(delta, wave, projector, snr_db, .1))
            if guarded and step % 20 == 0:
                guard_weight = float(np.clip(guard_weight + 5. * (float(guard) - .005), .5, 20.))
        if step % 20 == 0 or step == steps:
            record = {'step': step, 'content_index': index, 'loss': float(loss.detach()),
                'output_identity_margin': float(output_margin.detach()), 'guard': float(guard.detach()),
                'identity_preservation': float(preservation.detach()), 'content_relative_mse': float(content.detach()),
                'spectral_relative_l1': float(spectrum.detach()), 'guard_weight_next': guard_weight,
                'elapsed_s': round(time.monotonic() - started, 2)}
            trace.append(record)
            print(record, flush=True)
    with torch.no_grad():
        for _ in range(30):
            delta.copy_(constrain_delta(delta, wave, projector, snr_db, .1))
        result = (wave + delta).cpu().numpy()
    if callback:
        callback(result)
    return result, {'trace': trace, 'surrogate': 'ECAPA, participates in gradients',
        'held_out': ['GE2E', 'MOSNet', 'Whisper base', 'Whisper small.en'],
        'phase_gradient': f'last {gradient_iterations} GL iterations; preceding phases detached',
        'forward_gl_iterations': preprocess['n_iter'], 'guarded': guarded,
        'content_guard': 'AdaIN content latent and multi-resolution spectra, not a direct ASR constraint',
        'elapsed_s': round(time.monotonic() - started, 2)}
