"""Enrollment-discriminant reference defense; independent scorers stay held out.

This is a new exploratory method, not the paper's original distance objective.
The linear discriminant is fitted only to native AdaIN enrollment features.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .content_robust_recovery import normalized_gradient, remove_content_conflict
from .joint_recovery import spectral_consistency
from .masking import MaskingProjection
from .recovery import constrain_delta
from .staged_recovery import NativeFeatures


def fit_discriminant(features, labels, classes, shrinkage=.2):
    """Shrink within-class covariance; retain centered, equal-prior LDA scores."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (x.ndim != 2 or len(x) != len(y) or classes < 2 or not 0 < shrinkage <= 1
            or not np.isfinite(x).all() or set(y) != set(range(classes))):
        raise ValueError('Finite complete enrollment and positive shrinkage required')
    if any(np.sum(y == k) < 2 for k in range(classes)):
        raise ValueError('At least two enrollment sentences per class required')
    center = x.mean(0)
    means = np.stack([x[y == k].mean(0) - center for k in range(classes)])
    residual = x - center - means[y]
    cov = residual.T @ residual / (len(x) - classes)
    variance = max(float(np.trace(cov) / x.shape[1]), 1e-8)
    cov = (1 - shrinkage) * cov + shrinkage * variance * np.eye(x.shape[1])
    weight = np.linalg.solve(cov, means.T).T
    bias = -.5 * np.sum(means * weight, axis=1)
    logits = (x - center) @ weight.T + bias
    gaps = []
    for k in range(classes):
        selected = logits[y == k]
        gaps.extend(selected[:, k] - np.delete(selected, k, axis=1).max(1))
    scale = max(float(np.median(np.abs(gaps))), 1.)
    return {'center': center.astype('float32'), 'weight': (weight / scale).astype('float32'),
        'bias': (bias / scale).astype('float32'), 'scale': scale,
        'shrinkage': shrinkage, 'enrollment_accuracy': float(np.mean(logits.argmax(1) == y))}


class EnrollmentDiscriminant(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        for key in ('center', 'weight', 'bias'):
            self.register_buffer(key, torch.as_tensor(parameters[key], dtype=torch.float32))

    def forward(self, embedding):
        return (embedding - self.center) @ self.weight.T + self.bias


def class_margin(logits, own_index):
    if logits.shape[-1] < 2 or not 0 <= own_index < logits.shape[-1]:
        raise ValueError('Own class must occur in a multiclass enrollment')
    other = torch.cat((logits[..., :own_index], logits[..., own_index+1:]), -1)
    return logits[..., own_index] - other.max(-1).values


def quality_caps(baseline):
    return {'ctc_max': baseline['ctc'] + .01,
        'own_similarity_min': baseline['own_similarity'] - .005,
        'spectrum_max': baseline['spectrum'] + .03}


def acceptable(values, caps):
    return (all(np.isfinite(values[k]) for k in ('ctc', 'own_similarity', 'spectrum', 'defense'))
        and values['ctc'] <= caps['ctc_max']
        and values['own_similarity'] >= caps['own_similarity_min']
        and values['spectrum'] <= caps['spectrum_max'])


def make_projector(theta, psd_max, preprocess, mask_relax_db):
    if not np.isfinite(mask_relax_db) or not 0 <= mask_relax_db <= 6:
        raise ValueError('Mask allowance must be between zero and six dB')
    projector = MaskingProjection(theta, psd_max, **{k: preprocess[k] for k in
        ('n_fft', 'hop_length', 'win_length')}, margin_db=0.)
    projector.limit.mul_(10 ** (mask_relax_db / 20))
    return projector


def refine_discriminant(model, original, initial, theta, psd_max, attr, preprocess,
        discriminant, own_index, content_mels, ctc, transcript, full_original, interval,
        surrogate, *, steps=800, learning_rate=.0003, snr_db=20., mask_relax_db=0.,
        max_seconds=420., callback=None):
    """Move the worst native/decoded identity margin with fixed quality caps.

    Each block has 16 updates. Checkpoints are accepted solely by original CTC,
    ECAPA similarity, spectral distance and native training loss. Failed blocks
    restore the last accepted waveform and reduce the rate. No output scorer is
    called here; original ECAPA is an acceptance proxy, not independent evidence.
    """
    if (steps < 1 or learning_rate <= 0 or max_seconds <= 0 or not content_mels
            or snr_db < 10 or not 0 <= mask_relax_db <= 6):
        raise ValueError('Invalid fixed refinement configuration')
    device = next(model.parameters()).device
    wave, start, full = [torch.as_tensor(x, dtype=torch.float32, device=device)
        for x in (original, initial, full_original)]
    begin, end = map(int, interval)
    if start.shape != wave.shape or len(wave) != end-begin or not 0 <= begin < end <= len(full):
        raise ValueError('Full context and protected crop differ')
    features = NativeFeatures(preprocess, attr).to(device)
    projector = make_projector(theta, psd_max, preprocess, mask_relax_db).to(device)
    def restore(crop): return torch.cat((full[:begin], crop, full[end:]))
    with torch.no_grad():
        original_identity = surrogate(full).detach()
        contents = [model.content_encoder(m.to(device))[0].detach() for m in content_mels]

    def defense_loss(crop):
        identity = model.speaker_encoder(features(crop))
        margins = [class_margin(discriminant(identity), own_index).mean()]
        margins += [class_margin(discriminant(model.speaker_encoder(model.decoder(c, identity))),
            own_index).mean() for c in contents]
        # Smooth worst-case aggregation: all paths retain a gradient.
        vector = torch.stack(margins)
        return torch.logsumexp(vector / .25, dim=0) * .25

    def checks(crop):
        restored = restore(crop)
        return {'ctc': float(ctc.loss(restored, preprocess['sample_rate'], transcript)),
            'own_similarity': float(F.cosine_similarity(surrogate(restored), original_identity).mean()),
            'spectrum': float(spectral_consistency(wave, crop)), 'defense': float(defense_loss(crop))}

    started = time.monotonic()
    with torch.no_grad():
        baseline = checks(start); caps = quality_caps(baseline)
        accepted = (start-wave).clone(); delta = accepted.clone()
        best = baseline['defense']
    delta.requires_grad_(True)
    rate = learning_rate
    accepted_blocks = rejected_blocks = conflicts = 0
    trace = []
    for step in range(1, steps+1):
        if time.monotonic()-started > max_seconds: raise TimeoutError('Refinement time budget reached')
        crop = wave+delta
        defense = defense_loss(crop)
        content = ctc.loss(restore(crop), preprocess['sample_rate'], transcript)
        gd, = torch.autograd.grad(defense, delta)
        gc, = torch.autograd.grad(content, delta)
        if not torch.isfinite(gd).all() or not torch.isfinite(gc).all():
            raise ValueError('Nonfinite native-discriminant or CTC gradient')
        with torch.no_grad():
            direction = normalized_gradient(gd)
            conflicts += int(float((direction*gc).sum()) < 0)
            direction = remove_content_conflict(direction, gc)
            if float(content) > baseline['ctc']:
                direction += .25 * normalized_gradient(gc)
            direction = normalized_gradient(direction)
            delta.copy_(constrain_delta(delta-rate*direction, wave, projector, snr_db, .1))
            if step % 16 == 0 or step == steps:
                for _ in range(30): delta.copy_(constrain_delta(delta, wave, projector, snr_db, .1))
                values = checks(wave+delta)
                allowed = acceptable(values, caps) and values['defense'] < best - 1e-6
                if allowed:
                    accepted.copy_(delta); best=values['defense']; accepted_blocks += 1
                else:
                    delta.copy_(accepted); rejected_blocks += 1
                    rate=max(rate*.7, learning_rate/32)
                record = {'step': step, 'accepted': allowed, 'checks': values, 'rate_next': rate,
                    'elapsed_s': round(time.monotonic()-started, 2)}
                trace.append(record)
                if step % 160 == 0 or step == steps: print(record, flush=True)
    result = (wave+accepted).detach().cpu().numpy()
    if callback: callback(result)
    return result, {'baseline_proxy_checks': baseline, 'proxy_caps': caps,
        'accepted_blocks': accepted_blocks, 'rejected_blocks': rejected_blocks,
        'content_gradient_conflicts': conflicts, 'trace': trace,
        'elapsed_s': round(time.monotonic()-started, 2),
        'defense_surrogate': 'Enrollment LDA on native reference and decoded-mel speaker features',
        'independent_evaluators': ['GE2E', 'MOSNet', 'Whisper base', 'Whisper small.en'],
        'relaxation_note': 'Positive mask_relax_db permits excess over original masking cap; not an inaudibility guarantee'}


@torch.no_grad()
def export_checks(model, crop, changed, full, interval, attr, pp, discriminant,
        own_index, content_mels, ctc, text, surrogate):
    device=next(model.parameters()).device
    wave, altered, full = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in (crop, changed, full)]
    begin, end=map(int, interval)
    restored=torch.cat((full[:begin], altered, full[end:]))
    features=NativeFeatures(pp, attr).to(device)
    identity=model.speaker_encoder(features(altered))
    margins=[class_margin(discriminant(identity), own_index).mean()]
    for mel in content_mels:
        c=model.content_encoder(mel)[0]
        margins.append(class_margin(discriminant(model.speaker_encoder(model.decoder(c, identity))), own_index).mean())
    return {'ctc': float(ctc.loss(restored, pp['sample_rate'], text)),
        'own_similarity': float(F.cosine_similarity(surrogate(restored), surrogate(full)).mean()),
        'spectrum': float(spectral_consistency(wave, altered)),
        'defense': float(torch.logsumexp(torch.stack(margins)/.25, 0)*.25)}
