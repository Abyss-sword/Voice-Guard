"""Pinned upstream VQMIVC inference with its released ParallelWaveGAN vocoder.

Compatibility changes: keyword-only librosa calls, direct in-memory vocoding
instead of Kaldi temporary archives, and explicit paired vocoder noise seeds.
Network definitions and state dictionaries are loaded unchanged from upstream.
"""
from pathlib import Path
import sys
import importlib.util
import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from scipy import signal
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'third_party/vqmivc'


def logmel(wave, sample_rate=16000):
    spectrum=librosa.stft(np.asarray(wave,dtype=np.float32),n_fft=400,
        hop_length=160,win_length=400,window='hann',center=True,pad_mode='reflect')
    basis=librosa.filters.mel(sr=sample_rate,n_fft=400,n_mels=80,fmin=80,fmax=7600)
    return np.log10(np.maximum(1e-10,np.abs(spectrum).T@basis.T))


def load_source_module(name, filename):
    spec=importlib.util.spec_from_file_location(name,SOURCE/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class VQMIVC:
    def __init__(self, model_dir='checkpoints/vqmivc-official',device='cuda'):
        self.device=torch.device(device)
        self.model_dir=Path(model_dir)
        encoder=load_source_module('voiceguard_vqmivc_encoder','model_encoder.py')
        decoder=load_source_module('voiceguard_vqmivc_decoder','model_decoder.py')
        self.encoder=encoder.Encoder(in_channels=80,channels=512,n_embeddings=512,z_dim=64,c_dim=256)
        self.pitch_encoder=encoder.Encoder_lf0()
        self.speaker_encoder=encoder.SpeakerEncoder()
        self.decoder=decoder.Decoder_ac(dim_neck=64)
        checkpoint=torch.load(self.model_dir/'VQMIVC-model.ckpt-500.pt',map_location='cpu')
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.speaker_encoder.load_state_dict(checkpoint['encoder_spk'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        for model in (self.encoder,self.pitch_encoder,self.speaker_encoder,self.decoder):
            model.eval().requires_grad_(False).to(self.device)
        sys.path.insert(0,str(SOURCE/'ParallelWaveGAN'))
        # pyworld is installed to an isolated optional vendor path, never into
        # the frozen primary environment.
        sys.path.insert(0,str(ROOT/'outputs/vqmivc/vendor'))
        from parallel_wavegan.utils import load_model
        vocoder_dir=self.model_dir/'vocoder'
        config=yaml.safe_load((vocoder_dir/'config.yml').read_text())
        if config['sampling_rate']!=16000: raise ValueError('Expected upstream 16 kHz vocoder')
        # SciPy moved this window to signal.windows. The upstream PQMF import
        # binds the identical function; its source bytes remain untouched.
        with patch.object(signal,'kaiser',signal.windows.kaiser,create=True):
            self.vocoder=load_model(str(vocoder_dir/'checkpoint-3000000steps.pkl'),config)
        self.vocoder.remove_weight_norm()
        self.vocoder.eval().requires_grad_(False).to(self.device)
        self.stats=np.load(SOURCE/'mel_stats/stats.npy')
        self.cache={}

    def features(self,path,pitch=False):
        key=(str(Path(path).resolve()),pitch)
        if key in self.cache: return self.cache[key]
        wave,sr=sf.read(path,dtype='float64')
        if wave.ndim!=1 or not np.isfinite(wave).all(): raise ValueError('Finite mono WAV required')
        if sr!=16000:
            import resampy
            wave=resampy.resample(wave,sr,16000,axis=0)
        peak=np.max(np.abs(wave))
        if peak>1: wave=wave/peak
        mel=(logmel(wave)-self.stats[0])/(self.stats[1]+1e-8)
        value=torch.tensor(mel.T[None],dtype=torch.float32,device=self.device)
        lf0=None
        if pitch:
            import pyworld as pw
            f0,t=pw.dio(wave.astype('float64'),16000,frame_period=10.)
            f0=pw.stonemask(wave.astype('float64'),f0,t,16000)[:len(mel)].astype('float32')
            if len(f0)!=len(mel): raise ValueError('Pitch and mel lengths differ')
            voiced=f0!=0
            if voiced.any():
                f0[voiced]=np.log(f0[voiced])
                f0[voiced]=(f0[voiced]-f0[voiced].mean())/(f0[voiced].std()+1e-8)
            lf0=torch.tensor(f0[None],device=self.device)
        self.cache[key]=(value,lf0)
        return value,lf0

    @torch.inference_mode()
    def convert(self,source,reference,output,seed=2026110906):
        source_mel,lf0=self.features(source,pitch=True)
        reference_mel,_=self.features(reference)
        z,*_=self.encoder.encode(source_mel)
        decoded=self.decoder(z,self.pitch_encoder(lf0),self.speaker_encoder(reference_mel))
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices: torch.cuda.manual_seed_all(seed)
            wave=self.vocoder.inference(decoded.squeeze(0),normalize_before=False).reshape(-1)
        if not torch.isfinite(wave).all(): raise ValueError('Nonfinite vocoder output')
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        sf.write(output,wave.cpu().numpy(),16000,subtype='PCM_16')
        return {'samples':len(wave),'sample_rate':16000,'seed':seed,
            'abs_peak_before_pcm':float(wave.abs().max()),
            'clipped_fraction_before_pcm':float((wave.abs()>1).float().mean()),
            'normalize_before':False}
