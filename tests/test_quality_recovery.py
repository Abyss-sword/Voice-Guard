"""Check optimization gradients and noise budgets without independent evaluators."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from vguard.recovery import constrain_delta,recover_waveform
from vguard.masking import generate_th


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Conv1d(1,3,3,padding=1)
        self.condition=nn.Linear(3,1)
    def speaker_encoder(self,mel):return torch.tanh(self.encoder(mel)).mean(-1)
    def content_encoder(self,mel):return mel,torch.zeros_like(mel)
    def decoder(self,content,condition):return content+self.condition(condition).unsqueeze(-1)


class QualityRecoveryTests(unittest.TestCase):
    def test_constrained_noise_obeys_snr_peak_and_waveform_headroom(self):
        original=torch.linspace(-.99,.99,200)
        candidate=torch.randn(200)
        delta=constrain_delta(candidate,original,nn.Identity(),25,.05)
        self.assertLessEqual(float(delta.abs().max()),.05+1e-7)
        self.assertLessEqual(float((original+delta).abs().max()),1)
        self.assertLessEqual(float(delta.square().mean()),float(original.square().mean()*10**(-2.5))+1e-9)
        self.assertTrue(torch.equal(constrain_delta(candidate,torch.zeros_like(original),nn.Identity(),25,.05),torch.zeros_like(original)))

    def test_feedback_propagates_to_waveform_without_model_weight_updates(self):
        torch.manual_seed(4);model=TinyModel().eval().requires_grad_(False)
        original=(.2*np.sin(np.arange(600)*.13)).astype(np.float32)
        cfg={'n_fft':128,'hop_length':32,'win_length':128}
        theta,psd=generate_th(original,16000,**cfg)
        original_copy=original.copy();saved=[]
        with patch('vguard.recovery.wav2mel_tensor',side_effect=lambda wave,**kw:wave.reshape(1,1,-1)):
            wave,trace=recover_waveform(model,original,np.array([1,-1,.5],np.float32),
                [torch.from_numpy(original).reshape(1,1,-1)],theta,psd,{},cfg,
                steps=4,checkpoints=(4,),callback=lambda step,value:saved.append(value))
        self.assertTrue(np.isfinite(wave).all());self.assertEqual(len(saved),1)
        self.assertGreater(float(np.abs(wave-original).max()),0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        np.testing.assert_array_equal(original,original_copy)
        self.assertLessEqual(float(np.mean((wave-original)**2)),float(np.mean(original**2)*10**(-2.5))+1e-9)


if __name__=='__main__':unittest.main()
