from typing import Tuple
import numpy as np
import scipy.signal
import av


def read_audio_mono(mp4_path: str) -> Tuple[np.ndarray, int, float]:
    """
    Decode the audio track of a video as mono float32 at its native sample rate.

    Returns:
        audio: (N,) float32 array
        sample_rate: Hz
        audio_offset: start time of audio relative to the first video frame (sec)
    """
    with av.open(mp4_path) as container:
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=sample_rate)

        video_start = 0.0
        if video_stream.start_time is not None:
            video_start = float(video_stream.start_time * video_stream.time_base)

        audio_start = None
        chunks = list()
        for frame in container.decode(audio_stream):
            if audio_start is None:
                audio_start = float(frame.time)
            for mono_frame in resampler.resample(frame):
                chunks.append(mono_frame.to_ndarray()[0])
        for mono_frame in resampler.resample(None):
            chunks.append(mono_frame.to_ndarray()[0])
    assert audio_start is not None, f"No audio in {mp4_path}"

    audio = np.concatenate(chunks).astype(np.float32)
    audio_offset = audio_start - video_start
    return audio, sample_rate, audio_offset


def cross_correlation_lag(ref: np.ndarray, query: np.ndarray) -> Tuple[int, float]:
    """
    Full cross-correlation of two waveforms, as in freemocap/skelly_synchronize
    and evoth/gopro-video-sync.
    Sounds heard by only one microphone (e.g. a camera's own gripper)
    are incoherent between cameras and do not form a correlation peak.
    PHAT weighting (GCC-PHAT) is not used: GoPro audio has almost no
    acoustic energy above ~8kHz, and whitening those bins lets
    codec artifacts produce false peaks.

    Returns:
        lag: in samples, a sound at ref[n] appears at query[n + lag]
        standard_score: (peak - mean) / std of the correlation,
            the confidence measure reported by BBC audio-offset-finder
    """
    corr = scipy.signal.correlate(query, ref, mode='full', method='fft')
    lags = scipy.signal.correlation_lags(len(query), len(ref), mode='full')
    i = int(np.argmax(corr))
    standard_score = float((corr[i] - corr.mean()) / corr.std())
    return int(lags[i]), standard_score
