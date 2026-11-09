"""Development extension: native references and paper-style soft masking search.

This preserves earlier frozen optimizers. Search uses only native Es identity,
never the independent GE2E/MOS/ASR checks. It is an implementation interpretation,
not the unavailable complete paper training/evaluation pipeline.
"""
import torch
from torch.nn import functional as F
from .masking import Transform
from .staged_recovery import NativeFeatures


def soft_recover(model, original, target_embeddings, theta, psd_max, attr, preprocess,
                 prototypes, *, strong_steps=3000, conceal_steps=1500,
                 learning_rate=.001, searches=8, epsilon=.05, callback=None):
    if min(strong_steps, conceal_steps, searches) < 1 or epsilon <= 0 or learning_rate <= 0:
        raise ValueError('Positive steps, search count, epsilon and learning rate required')
    device = next(model.parameters()).device
    wave = torch.as_tensor(original, device=device, dtype=torch.float32)
    targets = torch.as_tensor(target_embeddings, device=device, dtype=torch.float32)
    bank = torch.as_tensor(prototypes, device=device, dtype=torch.float32)
    features = NativeFeatures(preprocess, attr).to(device)
    transform = Transform(**{k: preprocess[k] for k in ('n_fft', 'hop_length', 'win_length')}).to(device)
    threshold = torch.as_tensor(theta, device=device, dtype=torch.float32)
    with torch.no_grad():
        own = model.speaker_encoder(features(wave))
        target = targets.mean(0, keepdim=True)
        original_index = int(F.cosine_similarity(own, bank).argmax())
    def prediction(embedding):
        return int(F.cosine_similarity(embedding.detach(), bank).argmax())
    def audio(w):
        # Official fragment uses 2*eps*tanh(w), unlike Eq.3's eps*tanh(w).
        return (wave + 2 * epsilon * w.tanh()).clamp(-1, 1)
    def objective(embedding):
        return F.mse_loss(embedding, target) - .1 * F.mse_loss(embedding, own)
    w = torch.randn_like(wave).requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    trace = []
    for step in range(1, strong_steps + 1):
        embedding = model.speaker_encoder(features(audio(w)))
        loss = objective(embedding)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if w.grad is None or not torch.isfinite(w.grad).all(): raise ValueError('Invalid gradient')
        optimizer.step()
        if step % 500 == 0 or step == strong_steps:
            item = {'phase': 'strong', 'step': step, 'loss': float(loss.detach())}
            trace.append(item); print(item, flush=True)
    strong_w = w.detach().clone()
    with torch.no_grad():
        best_wave = audio(strong_w).detach().clone()
        strong_prediction = prediction(model.speaker_encoder(features(best_wave)))
    if callback: callback('strong', best_wave.cpu().numpy())
    alpha, low, high, chosen_alpha = 1., 0., None, None
    # Eight explicit attempts. Each starts from the same strong solution. Stop an
    # attempt when native identity is lost, retaining the last fully successful
    # attempt. Unlike the public while fragment, no double increment is copied.
    for attempt in range(searches):
        w = strong_w.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        success = False
        for step in range(1, conceal_steps + 1):
            proposed = audio(w)
            embedding = model.speaker_encoder(features(proposed))
            if prediction(embedding) == original_index: break
            masking = F.relu(transform(proposed - wave, psd_max) - threshold).mean()
            loss = objective(embedding) + alpha * masking
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if w.grad is None or not torch.isfinite(w.grad).all(): raise ValueError('Invalid gradient')
            optimizer.step()
        with torch.no_grad():
            proposed = audio(w)
            final_prediction = prediction(model.speaker_encoder(features(proposed)))
            success = step == conceal_steps and final_prediction != original_index
            final_mask = float(F.relu(transform(proposed-wave, psd_max)-threshold).mean())
        trace.append({'phase': 'conceal', 'attempt': attempt + 1, 'alpha': alpha,
            'steps': step, 'native_success': success, 'native_prediction': final_prediction,
            'mask_loss': final_mask})
        print(trace[-1], flush=True)
        if success:
            if chosen_alpha is None or alpha > chosen_alpha:
                chosen_alpha = alpha; best_wave = proposed.detach().clone()
            low = alpha; alpha = 2*alpha if high is None else (low + high)/2
        else:
            high = alpha; alpha = (low + high)/2
    result = best_wave.cpu().numpy()
    if callback: callback('final', result)
    return result, {'trace': trace, 'selected_alpha': chosen_alpha,
        'strong_native_prediction': strong_prediction, 'original_native_prediction': original_index,
        'fallback_to_strong': chosen_alpha is None, 'epsilon': epsilon,
        'amplitude_parameterization': '2*epsilon*tanh(w), clipped to waveform headroom',
        'hard_snr_constraint': False, 'evaluators_used_in_gradient_loss': False}
