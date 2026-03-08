"""
Audio utility functions for loading, saving, and processing audio.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def load_audio(
    path: str,
    sample_rate: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file and return a tensor with the given sample rate.

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate. The audio is resampled if needed.
        mono: If True, convert to mono by averaging channels.
        duration: If given, load at most this many seconds of audio.
        offset: Start reading audio at this offset (seconds).

    Returns:
        Tuple of (waveform tensor of shape (channels, samples), sample_rate).
    """
    import torchaudio

    # Load the full file (or just the header) first to get the native sample rate,
    # then compute offset and duration in native frames.
    info = torchaudio.info(path)
    native_sr = info.sample_rate
    frame_offset = int(offset * native_sr) if offset else 0
    num_frames = int(duration * native_sr) if duration is not None else -1
    waveform, sr = torchaudio.load(
        path,
        frame_offset=frame_offset,
        num_frames=num_frames if num_frames > 0 else -1,
    )
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int = 22050) -> None:
    """Save a waveform tensor to an audio file.

    Args:
        path: Output file path (format inferred from extension).
        waveform: Tensor of shape (channels, samples) or (samples,).
        sample_rate: Sample rate of the waveform.
    """
    import torchaudio

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform, sample_rate)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """Compute a log-mel spectrogram from a waveform tensor.

    Args:
        waveform: Tensor of shape (channels, samples) or (samples,).
        sample_rate: Sample rate of the waveform.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        n_mels: Number of mel filter banks.
        f_min: Minimum frequency for mel filters.
        f_max: Maximum frequency for mel filters. Defaults to sample_rate / 2.

    Returns:
        Log-mel spectrogram tensor of shape (channels, n_mels, time_frames).
    """
    import torchaudio.transforms as T

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max if f_max is not None else sample_rate // 2,
    )
    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(mel_spec.clamp(min=1e-5))
    return log_mel_spec


def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad or trim a waveform to a fixed length along the last dimension.

    Args:
        waveform: Tensor of shape (..., samples).
        target_length: Target number of samples.

    Returns:
        Tensor of shape (..., target_length).
    """
    current_length = waveform.shape[-1]
    if current_length == target_length:
        return waveform
    if current_length > target_length:
        return waveform[..., :target_length]
    pad_amount = target_length - current_length
    return torch.nn.functional.pad(waveform, (0, pad_amount))
