"""Inference port of lochenchou/MOSNet's original CNN-BLSTM checkpoint.

Matches the H5 model's Keras 2.1.2 configuration: TensorFlow SAME padding,
frequency/channel flatten order, hard-sigmoid recurrent gates, no batch norm.
Use tools/verify_original_mosnet.py to compare against TensorFlow on fixed inputs.
This is an additional evaluator, not a silent replacement for historical scores.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEIGHT_SHA256='78b75e7d76ee6074ea7d57dcffa56d0c90be9d3d8dedc2217e25e259423cb756'


class OriginalMOSNet(nn.Module):
    def __init__(self,checkpoint='checkpoints/mosnet-original/cnn_blstm.h5',device='cpu'):
        super().__init__()
        import h5py
        path=Path(checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=WEIGHT_SHA256:
            raise ValueError('Original MOSNet checkpoint SHA256 mismatch')
        with h5py.File(path,'r') as handle:
            weights=handle['model_weights']
            self.convs=nn.ModuleList()
            for index in range(1,13):
                values=weights[f'conv2d_{index}/conv2d_{index}']
                kernel=np.asarray(values['kernel:0'])
                conv=nn.Conv2d(kernel.shape[2],kernel.shape[3],3,stride=(1,3 if index%3==0 else 1))
                conv.weight.data.copy_(torch.from_numpy(kernel.transpose(3,2,0,1)))
                conv.bias.data.copy_(torch.from_numpy(np.asarray(values['bias:0'])))
                self.convs.append(conv)
            for direction in ('forward','backward'):
                group=weights[f'bidirectional_1/bidirectional_1/{direction}_lstm_1']
                for name in ('kernel','recurrent_kernel','bias'):
                    self.register_buffer(f'{direction}_{name}',torch.from_numpy(np.asarray(group[name+':0'])))
            self.dense=nn.Linear(256,128)
            self.frame=nn.Linear(128,1)
            for layer,name in ((self.dense,'time_distributed_2'),(self.frame,'Frame_score')):
                group=weights[f'{name}/{name}']
                layer.weight.data.copy_(torch.from_numpy(np.asarray(group['kernel:0']).T))
                layer.bias.data.copy_(torch.from_numpy(np.asarray(group['bias:0'])))
        self.to(device).eval().requires_grad_(False)

    def recurrent(self,sequence,direction):
        kernel=getattr(self,direction+'_kernel')
        recurrent=getattr(self,direction+'_recurrent_kernel')
        bias=getattr(self,direction+'_bias')
        hidden=sequence.new_zeros((sequence.shape[0],128));cell=hidden.clone()
        projected=sequence@kernel+bias
        steps=range(sequence.shape[1]) if direction=='forward' else range(sequence.shape[1]-1,-1,-1)
        outputs=[]
        for step in steps:
            gates=projected[:,step]+hidden@recurrent
            ingate,forget,candidate,outgate=gates.chunk(4,dim=-1)
            ingate=(.2*ingate+.5).clamp(0,1)
            forget=(.2*forget+.5).clamp(0,1)
            outgate=(.2*outgate+.5).clamp(0,1)
            cell=forget*cell+ingate*candidate.tanh()
            hidden=outgate*cell.tanh();outputs.append(hidden)
        if direction=='backward':outputs.reverse()
        return torch.stack(outputs,dim=1)

    def frame_scores(self,spectrogram):
        value=spectrogram.unsqueeze(1)
        for layer in self.convs:
            height,width=value.shape[-2:]
            sh,sw=layer.stride
            ph=max((math.ceil(height/sh)-1)*sh+3-height,0)
            pw=max((math.ceil(width/sw)-1)*sw+3-width,0)
            value=F.relu(layer(F.pad(value,(pw//2,pw-pw//2,ph//2,ph-ph//2))))
        value=value.permute(0,2,3,1).reshape(value.shape[0],value.shape[2],512)
        value=torch.cat([self.recurrent(value,direction) for direction in ('forward','backward')],dim=-1)
        return self.frame(F.relu(self.dense(value))).squeeze(-1)

    def forward(self,spectrogram):
        return self.frame_scores(spectrogram).mean(dim=1)

    @torch.inference_mode()
    def score(self,wave,sample_rate):
        import librosa
        from scipy.signal.windows import hamming
        wave=np.asarray(wave,dtype=np.float32)
        if wave.ndim!=1 or not wave.size or not np.isfinite(wave).all():
            raise ValueError('Finite nonempty mono waveform required')
        if sample_rate!=16000:wave=librosa.resample(wave,orig_sr=sample_rate,target_sr=16000)
        magnitude=np.abs(librosa.stft(wave,n_fft=512,hop_length=256,win_length=512,
            window=hamming(512,sym=True),center=True,pad_mode='reflect')).T.astype(np.float32)
        return float(self(torch.from_numpy(magnitude).unsqueeze(0).to(next(self.parameters()).device))[0])
