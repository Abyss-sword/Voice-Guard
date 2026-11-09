"""Development extension: retain baseline output disruption on two surrogates.

The output anchors are proxy constraints, not guarantees about independent GE2E.
This module leaves the first joint experiment's frozen implementation unchanged.
"""
from __future__ import annotations
import time
import librosa
import numpy as np
import torch
from torch.nn import functional as F

from .joint_recovery import WaveformReconstructor, identity_margin, spectral_consistency
from .masking import MaskingProjection
from .recovery import constrain_delta
from .staged_recovery import NativeFeatures


def refine_anchored(model, original, initial, content_mels, theta, psd_max, attr, preprocess,
                    surrogate, registry, own_index, *, steps=240, learning_rate=.00015,
                    snr_db=20., guarded=True, gradient_iterations=2, max_seconds=600.,
                    callback=None, native_prototypes=None):
    if not guarded or steps < 1 or max_seconds <= 0 or not content_mels or learning_rate <= 0:
        raise ValueError('Anchored extension requires guarded finite optimization')
    device = next(model.parameters()).device
    wave = torch.as_tensor(original, dtype=torch.float32, device=device)
    start = torch.as_tensor(initial, dtype=torch.float32, device=device)
    if start.shape != wave.shape or not torch.isfinite(start).all(): raise ValueError('Invalid warm start')
    registry = F.normalize(torch.as_tensor(registry, dtype=torch.float32, device=device), dim=-1)
    if native_prototypes is None:
        with np.load('outputs/staged_recovery/2026-11-09/native-target-bank.npz') as bank:
            native_prototypes = bank['embeddings'].copy()
    native = F.normalize(torch.as_tensor(native_prototypes, dtype=torch.float32, device=device), dim=-1)
    if len(native) != len(registry) or not 0 <= own_index < len(registry):
        raise ValueError('Native and ECAPA enrollment order must match')
    features = NativeFeatures(preprocess, attr).to(device)
    reconstructor = WaveformReconstructor(preprocess, attr, gradient_iterations=gradient_iterations).to(device)
    projector = MaskingProjection(theta, psd_max, **{k: preprocess[k] for k in
        ('n_fft', 'hop_length', 'win_length')}, margin_db=0.).to(device)

    def native_output(waveform):
        # Match VC's trimming on actual synthesized audio. Slice boundaries are
        # recomputed on detached audio; gradients pass through the chosen crop.
        _, interval = librosa.effects.trim(waveform.detach().cpu().numpy(), top_db=preprocess['top_db'])
        cropped = waveform[int(interval[0]):int(interval[1])]
        return F.normalize(model.speaker_encoder(features(cropped)), dim=-1)

    started = time.monotonic()
    with torch.no_grad():
        clean_features = features(wave)
        clean_content = model.content_encoder(clean_features)[0].detach()
        content_scale = clean_content.square().mean().clamp_min(1e-5)
        own_embedding = surrogate(wave).detach()
        contents = [model.content_encoder(value.to(device))[0].detach() for value in content_mels]
        delta = start - wave
        for _ in range(30): delta = constrain_delta(delta, wave, projector, snr_db, .1)
        baseline_identity = model.speaker_encoder(features(wave + delta))
        anchors = []
        for content in contents:
            output = reconstructor(model.decoder(content, baseline_identity))
            ecapa = surrogate(output)
            native_embedding = native_output(output)
            anchors.append({'margin': min(-.12, float(identity_margin(ecapa, registry, own_index)) - .02),
                'ecapa_own_similarity': float((ecapa @ registry[own_index]).mean()) - .08,
                'native_own_similarity': float((native_embedding @ native[own_index]).mean()) - .08})
    delta = delta.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    guard_weight = 2.
    trace = []
    for step in range(1, steps + 1):
        if time.monotonic() - started > max_seconds: raise TimeoutError('Anchored per-utterance time limit')
        protected = wave + delta
        mel = features(protected)
        index = (step - 1) % len(contents)
        converted = reconstructor(model.decoder(contents[index], model.speaker_encoder(mel)))
        output_embedding = surrogate(converted)
        margin = identity_margin(output_embedding, registry, own_index).mean()
        own_similarity = (output_embedding @ registry[own_index]).mean()
        native_similarity = (native_output(converted) @ native[own_index]).mean()
        anchor = anchors[index]
        defense = F.relu(margin - anchor['margin'])
        defense = defense + .5 * F.relu(own_similarity - anchor['ecapa_own_similarity'])
        defense = defense + .5 * F.relu(native_similarity - anchor['native_own_similarity'])
        protected_embedding = surrogate(protected)
        preservation = F.relu(.98 - F.cosine_similarity(protected_embedding, own_embedding)).mean()
        preservation = preservation + F.relu(.03 - identity_margin(protected_embedding, registry, own_index)).mean()
        content_loss = F.mse_loss(model.content_encoder(mel)[0], clean_content) / content_scale
        spectrum = spectral_consistency(wave, protected)
        guard = preservation + .2 * F.relu(content_loss - .02) + .1 * F.relu(spectrum - .08)
        loss = defense + guard_weight * guard
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all(): raise ValueError('Invalid anchored gradient')
        torch.nn.utils.clip_grad_norm_([delta], 1.)
        optimizer.step()
        with torch.no_grad():
            delta.copy_(constrain_delta(delta, wave, projector, snr_db, .1))
            if step % 20 == 0:
                guard_weight = float(np.clip(guard_weight + 5. * (float(guard) - .005), .5, 20.))
        if step % 20 == 0 or step == steps:
            record = {'step': step, 'content_index': index, 'loss': float(loss.detach()),
                'output_margin': float(margin.detach()), 'output_own_similarity': float(own_similarity.detach()),
                'native_output_own_similarity': float(native_similarity.detach()),
                'defense': float(defense.detach()), 'guard': float(guard.detach()),
                'content_relative_mse': float(content_loss.detach()),
                'spectral_relative_l1': float(spectrum.detach()), 'guard_weight_next': guard_weight,
                'elapsed_s': round(time.monotonic() - started, 2)}
            trace.append(record); print(record, flush=True)
    with torch.no_grad():
        for _ in range(30): delta.copy_(constrain_delta(delta, wave, projector, snr_db, .1))
        result = (wave + delta).cpu().numpy()
    if callback: callback(result)
    return result, {'trace': trace, 'anchors': anchors,
        'surrogates': ['ECAPA', 'native AdaIN speaker encoder on reconstructed, trimmed waveform'],
        'held_out': ['GE2E', 'MOSNet', 'Whisper base', 'Whisper small.en'],
        'phase_gradient': 'Only last 2 GL iterations differentiated; detached crop boundaries',
        'constraint_note': 'Relative-output targets are soft proxy constraints, not independent-quality guarantees',
        'elapsed_s': round(time.monotonic() - started, 2)}
