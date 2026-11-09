"""Experimental constrained waveform optimization, separate from the published VG port.

Only AdaIN-VC is differentiated. ECAPA, MOSNet and Whisper remain independent.
The feedback loss acts on decoded mel features, not a differentiable waveform vocoder.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .features import wav2mel_tensor
from .masking import MaskingProjection


def constrain_delta(delta, original, projector, snr_db, peak):
    """Apply masking projection, a global noise budget, and waveform headroom."""
    if not math.isfinite(snr_db) or not math.isfinite(peak) or peak <= 0:
        raise ValueError('Finite SNR and positive peak required')
    delta = projector(delta).clamp(-peak, peak)
    delta = torch.maximum(torch.minimum(delta, 1-original), -1-original)
    budget = original.square().mean().sqrt() * 10**(-snr_db/20)
    gain = (budget / delta.square().mean().sqrt().clamp_min(1e-20)).clamp(max=1)
    return delta * gain


def recover_waveform(model, original, target_embedding, content_mels, theta, psd_max,
                     attr, feature_kwargs, *, objective='feedback', steps=1500,
                     learning_rate=.0003, snr_db=25., margin_db=3., peak=.1,
                     initial_delta=None, checkpoints=(500,1500), callback=None):
    if objective not in ('embedding','feedback') or steps < 1 or learning_rate <= 0:
        raise ValueError('Invalid objective or optimization settings')
    if objective == 'feedback' and not content_mels:
        raise ValueError('Feedback requires fixed content inputs')
    device=next(model.parameters()).device
    original=torch.as_tensor(original,dtype=torch.float32,device=device)
    target_embedding=torch.as_tensor(target_embedding,dtype=torch.float32,device=device).reshape(1,-1)
    projector=MaskingProjection(theta,psd_max,**{key:feature_kwargs[key] for key in
        ('n_fft','hop_length','win_length')},margin_db=margin_db).to(device)
    def features(wave):return wav2mel_tensor(wave,attr=attr,**feature_kwargs)
    with torch.no_grad():
        clean_mel=features(original)
        clean_embedding=model.speaker_encoder(clean_mel)
        contents=[model.content_encoder(mel.to(device))[0].detach() for mel in content_mels]
        clean_feedback=[model.speaker_encoder(model.decoder(mu,clean_embedding)).detach() for mu in contents]
        target_feedback=[model.speaker_encoder(model.decoder(mu,target_embedding)).detach() for mu in contents]
        delta=(torch.randn_like(original)*1e-4 if initial_delta is None else
               torch.as_tensor(initial_delta,dtype=torch.float32,device=device).clone())
        for _ in range(30):delta=constrain_delta(delta,original,projector,snr_db,peak)
    delta=delta.detach().requires_grad_(True)
    optimizer=torch.optim.Adam([delta],lr=learning_rate)
    trace=[]
    for step in range(1,steps+1):
        embedding=model.speaker_encoder(features(original+delta))
        direct=F.cosine_similarity(embedding,clean_embedding).mean()-F.cosine_similarity(embedding,target_embedding).mean()
        if objective=='feedback':
            index=(step-1)%len(contents)
            decoded_embedding=model.speaker_encoder(model.decoder(contents[index],embedding))
            feedback=F.cosine_similarity(decoded_embedding,clean_feedback[index]).mean()-F.cosine_similarity(decoded_embedding,target_feedback[index]).mean()
            loss=feedback+.25*direct
        else:
            feedback=direct.new_zeros(())
            loss=direct
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise ValueError('Nonfinite or missing waveform gradient')
        optimizer.step()
        with torch.no_grad():delta.copy_(constrain_delta(delta,original,projector,snr_db,peak))
        if step%100==0 or step==steps:
            record={'step':step,'loss':float(loss.detach()),'direct':float(direct.detach()),
                    'feedback':float(feedback.detach()),'noise_rms':float(delta.detach().square().mean().sqrt())}
            trace.append(record);print(record,flush=True)
        if step in checkpoints or step==steps:
            with torch.no_grad():
                exported=delta.clone()
                for _ in range(30):exported=constrain_delta(exported,original,projector,snr_db,peak)
                wave=(original+exported).cpu().numpy()
            if callback is not None:callback(step,wave)
    return wave,trace
