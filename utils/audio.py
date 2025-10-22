import subprocess
import matplotlib

matplotlib.use('Agg')
import librosa
import librosa.filters
import torch
import numpy as np
from scipy import signal
from scipy.io import wavfile
from modules.nsf_hifigan.nvSTFT import STFT


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav *= 32767
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def get_hop_size(hparams):
    hop_size = hparams['hop_size']
    if hop_size is None:
        assert hparams['frame_shift_ms'] is not None
        hop_size = int(hparams['frame_shift_ms'] / 1000 * hparams['audio_sample_rate'])
    return hop_size


###########################################################################################
def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams['fft_size'], hop_length=get_hop_size(hparams),
                        win_length=hparams['win_size'], pad_mode='constant')


def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams['win_size'])


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


# Conversions
def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def normalize(S, hparams):
    return (S - hparams['min_level_db']) / -hparams['min_level_db']

def get_mel_torch(waveform, sample_rate, *, num_mel_bins=128, hop_size=512, win_size=2048, fft_size=2048, fmin=40, fmax=16000, keyshift=0, speed=1, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    stft = STFT(sample_rate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax)
    with torch.no_grad():
        wav_torch = torch.from_numpy(waveform).to(device)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(0), keyshift=keyshift, speed=speed).squeeze(0).T

        return mel_torch.cpu().numpy()