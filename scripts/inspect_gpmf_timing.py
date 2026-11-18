"""
Timing of the GPMF sensor streams of GoPro videos against the camera clock
(STMP, microseconds) and the MP4 timeline (umi.common.gpmf_util).

Per stream, a line fit over all payloads:
  STMP = start + period * (samples before the payload, TSMP - n)
gives the stream's sample period on the camera clock, its start (first
sample) and the timing residual (RMS / max, us). Frame-rate streams
(CORI, MSKP) give the video frames on the camera clock; compared with the
MP4 video timeline (frame index * nominal frame duration) this shows whether
the video timeline runs on the camera clock. MSKP / LSKP: frames skipped.

python scripts/inspect_gpmf_timing.py <video.mp4> [<video.mp4> ...]
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# %%
import av
import click
import numpy as np
from umi.common.gpmf_util import stream_timing, mp4_gpmf_payloads

KEYS = ('ACCL', 'GYRO', 'CORI', 'MSKP', 'SHUT', 'AALP')


def fit_stream(d: dict) -> dict:
    before = d['tsmp'] - d['n']
    ok = d['stmp_us'] >= 0
    period, start = np.polyfit(before[ok], d['stmp_us'][ok], 1)
    residual = d['stmp_us'][ok] - (start + period * before[ok])
    return {'period_us': period, 'start_us': start, 'rms_us': float(np.sqrt(np.mean(residual ** 2))),
        'max_us': float(np.max(np.abs(residual))), 'n_payloads': int(ok.sum()), 'n_samples': int(d['tsmp'][-1])}


# %%
@click.command()
@click.argument('videos', nargs=-1, required=True)
def main(videos):
    for path in videos:
        with av.open(path) as container:
            v = container.streams.video[0]
            a = container.streams.audio[0]
            frame_duration = float(1 / v.average_rate)
            n_frames = v.frames
            video_duration = float(v.duration * v.time_base)
            audio_duration = float(a.duration * a.time_base)
            audio_rate = a.rate
        print(f"{path}")
        print(f"  MP4: {n_frames} frames x {frame_duration * 1e6:.3f} us = {video_duration:.4f} s video, "
            f"audio {audio_duration:.4f} s at {audio_rate} Hz")
        timing = stream_timing(path, KEYS)
        fits = {k: fit_stream(d) for k, d in timing.items()}
        for k, f in fits.items():
            print(f"  {k}: period {f['period_us']:.4f} us ({1e6 / f['period_us']:.4f} Hz), first sample at camera "
                f"{f['start_us'] / 1e3:.3f} ms, residual RMS {f['rms_us']:.1f} us max {f['max_us']:.1f} us, "
                f"{f['n_samples']} samples in {f['n_payloads']} payloads")
        # video frames on the camera clock vs the MP4 timeline
        frames = 'MSKP' if 'MSKP' in fits else 'CORI'
        if frames in fits:
            f = fits[frames]
            ppm = (f['period_us'] / (frame_duration * 1e6) - 1) * 1e6
            print(f"  video frames ({frames}): camera clock frame period / MP4 frame duration - 1 = {ppm:+.2f} ppm "
                f"({ppm * 60e-3:+.3f} ms/min); {f['n_samples']} frame samples vs {n_frames} MP4 frames")
            for k in ('ACCL', 'GYRO'):
                if k in fits:
                    print(f"  {k} first sample relative to the first video frame: "
                        f"{(fits[k]['start_us'] - f['start_us']) / 1e3:+.3f} ms (camera clock)")
        if 'MSKP' in timing:
            skips = list()
            for payload in mp4_gpmf_payloads(path):
                for devc in payload['klv']:
                    for strm in devc.get('children', []):
                        for c in strm.get('children', []):
                            if c['key'] in ('MSKP', 'LSKP') and 'value' in c:
                                skips.append((c['key'], np.count_nonzero(c['value'])))
            for key in ('MSKP', 'LSKP'):
                nonzero = sum(n for k, n in skips if k == key)
                print(f"  {key}: {nonzero} nonzero frame-skip values")

# %%
if __name__ == "__main__":
    main()
