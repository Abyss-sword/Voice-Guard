"""Native-feature, two-stage recovery experiments; frozen older methods stay intact.

The VC model alone supplies gradients. External ASV is used only to select a target
identity from enrollment, never to differentiate through evaluation waveforms.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import librosa
from .recovery import constrain_delta
from .masking import MaskingProjection


class NativeFeatures(nn.Module):
    """Match AdaIN-VC get_spectrograms on an already trimmed waveform."""
    def __init__(self, preprocess, attr):
        super().__init__()
        self.pp = preprocess
        self.register_buffer('window', torch.hann_window(preprocess['win_length']))
        self.register_buffer('basis', torch.from_numpy(librosa.filters.mel(
            sr=preprocess['sample_rate'], n_fft=preprocess['n_fft'], n_mels=preprocess['n_mels'])))
        self.register_buffer('mean', torch.as_tensor(attr['mean'], dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(attr['std'], dtype=torch.float32))

    def forward(self, wave):
        pp = self.pp
        emphasized = torch.cat((wave[:1], wave[1:] - pp['preemph'] * wave[:-1]))
        spectrum = torch.stft(emphasized, n_fft=pp['n_fft'], hop_length=pp['hop_length'],
            win_length=pp['win_length'], window=self.window, center=True,
            pad_mode='reflect', return_complex=True)
        mel = self.basis @ spectrum.abs()
        mel = 20 * torch.log10(mel.clamp_min(1e-5))
        mel = ((mel - pp['ref_db'] + pp['max_db']) / pp['max_db']).clamp(1e-8, 1)
        return ((mel.T - self.mean) / self.std).T.unsqueeze(0)


def staged_recover(model, original, target_embeddings, content_mels, theta, psd_max,
                   attr, preprocess, *, objective='embedding', strong_steps=3000,
                   conceal_steps=1500, strong_snr_db=15., final_snr_db=20.,
                   learning_rate=.001, final_margin_db=0., callback=None):
    """Warm up without masking, then gradually tighten the psychoacoustic cap.

    MSE follows the paper's .1 repulsion term. The e2e extension also retains a
    quarter of normalized embedding loss. No ASV/MOS/ASR evaluator is used here.
    """
    if objective not in ('embedding', 'feedback', 'e2e'):
        raise ValueError('Unknown staged objective')
    if min(strong_steps, conceal_steps) < 1 or learning_rate <= 0 or final_snr_db < strong_snr_db:
        raise ValueError('Invalid stage configuration')
    device = next(model.parameters()).device
    wave = torch.as_tensor(original, dtype=torch.float32, device=device)
    targets = torch.as_tensor(target_embeddings, dtype=torch.float32, device=device)
    if targets.ndim != 2 or not torch.isfinite(targets).all():
        raise ValueError('Finite target embedding bank required')
    if objective != 'embedding' and not content_mels:
        raise ValueError('Converted objectives require content')
    features = NativeFeatures(preprocess, attr).to(device)
    projector = MaskingProjection(theta, psd_max, **{k: preprocess[k] for k in
        ('n_fft', 'hop_length', 'win_length')}, margin_db=final_margin_db).to(device)
    final_limit = projector.limit.clone()
    with torch.no_grad():
        own = model.speaker_encoder(features(wave))
        target = targets.mean(0, keepdim=True)
        embedding_scale = F.mse_loss(own, target).clamp_min(1e-5)
        contents = [model.content_encoder(m.to(device))[0].detach() for m in content_mels]
        clean_decoded = [model.decoder(c, own).detach() for c in contents]
        target_decoded = [torch.stack([model.decoder(c, t.unsqueeze(0)) for t in targets]).mean(0).detach() for c in contents]
        decoded_scale = [F.mse_loss(a, b).clamp_min(1e-5) for a, b in zip(clean_decoded, target_decoded)]
        feedback_scale = [F.mse_loss(model.speaker_encoder(c), target).clamp_min(1e-5) for c in clean_decoded]
        initial = torch.randn_like(wave) * 1e-4
    delta = initial.detach().requires_grad_(True)
    traces = []
    def direct_loss(embedding):
        return (F.mse_loss(embedding, target) - .1 * F.mse_loss(embedding, own)) / embedding_scale
    for phase, steps in [('strong', strong_steps), ('conceal', conceal_steps)]:
        optimizer = torch.optim.Adam([delta], lr=learning_rate if phase == 'strong' else learning_rate * .3)
        for step in range(1, steps + 1):
            embedding = model.speaker_encoder(features(wave + delta))
            direct = direct_loss(embedding)
            auxiliary = direct.new_zeros(())
            if phase == 'conceal' and objective != 'embedding':
                index = (step - 1) % len(contents)
                decoded = model.decoder(contents[index], embedding)
                if objective == 'e2e':
                    auxiliary = (F.mse_loss(decoded, target_decoded[index]) - .1 * F.mse_loss(decoded, clean_decoded[index])) / decoded_scale[index]
                else:
                    feedback = model.speaker_encoder(decoded)
                    auxiliary = (F.mse_loss(feedback, target) - .1 * F.mse_loss(feedback, own)) / feedback_scale[index]
                loss = auxiliary + .25 * direct
            else:
                loss = direct
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if delta.grad is None or not torch.isfinite(delta.grad).all():
                raise ValueError('Invalid waveform gradient')
            optimizer.step()
            with torch.no_grad():
                if phase == 'strong':
                    delta.copy_(constrain_delta(delta, wave, nn.Identity(), strong_snr_db, .1))
                else:
                    progress = min(1., step / max(1, int(steps * 2/3)))
                    projector.limit.copy_(final_limit * 10 ** (12 * (1 - progress) / 20))
                    snr = strong_snr_db + (final_snr_db - strong_snr_db) * progress
                    delta.copy_(constrain_delta(delta, wave, projector, snr, .1))
            if step % 250 == 0 or step == steps:
                record = {'phase': phase, 'step': step, 'loss': float(loss.detach()),
                          'embedding_loss': float(direct.detach()), 'auxiliary_loss': float(auxiliary.detach()),
                          'noise_rms': float(delta.detach().square().mean().sqrt())}
                traces.append(record)
                print(record, flush=True)
        if phase == 'strong' and callback:
            callback('strong', (wave + delta).detach().cpu().numpy())
    with torch.no_grad():
        projector.limit.copy_(final_limit)
        for _ in range(30): delta.copy_(constrain_delta(delta, wave, projector, final_snr_db, .1))
        result = (wave + delta).cpu().numpy()
        exported_embedding_loss = float(direct_loss(model.speaker_encoder(features(wave + delta))))
    if callback: callback('final', result)
    return result, {'trace': traces, 'exported_embedding_loss': exported_embedding_loss,
                    'evaluators_used_in_gradient_loss': False,
                    'target_embedding_norms': targets.norm(dim=1).cpu().tolist()}
