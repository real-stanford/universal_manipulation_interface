"""
Check episode segmentation (scripts_tag_tracking/00b_segment_episodes.py) of
continuous takes: the same detection (umi.tag_tracking.episodes.segment_take),
with plots and per-bang details for tuning the 'episodes' config settings.

Reads, per head camera task video: wrist demos/*/imu_data.json,
audio_sync_offsets.json, head demos/*/tag_corners.pkl, the head video's audio.
Writes <session>/tag_tracking/wrist_events/<head video>.png and prints
signal statistics and, for the first / last REPORT_WINDOW_SEC of each stretch
between idle periods, every joint wrist spike: time strength/dt/head audio
click[/dock], * = taken as bang.

Plot, per 2 minutes: top panel gyro RMS of both grippers (log scale) with the
still threshold; green = idle (both still, dock confirmed by the head view),
orange = both still but no docked view; blue band = episode; red lines =
bangs; black ticks at the top = head frames with the docked view.
Middle: acceleration spikes of both grippers with bang_accel.
Bottom: head audio click ratio with audio_ratio.

python scripts_tag_tracking/check_wrist_events.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import importlib
import pathlib
import click
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.episodes import segment_take

segment_step = importlib.import_module('scripts_tag_tracking.00b_segment_episodes')
mmss = segment_step.mmss

ROW_SEC = 120.0
REPORT_WINDOW_SEC = 5.0


def plot_take(path, title, names, params, result):
    d = result['diagnostics']
    grid = d['grid']
    n_rows = int(np.ceil(grid[-1] / ROW_SEC))
    fig, axes = plt.subplots(3 * n_rows, 1, figsize=(20, 5.6 * n_rows), squeeze=False,
        gridspec_kw={'height_ratios': [3, 2, 1.5] * n_rows})
    for r in range(n_rows):
        ax, ax_accel, ax_audio = axes[3 * r, 0], axes[3 * r + 1, 0], axes[3 * r + 2, 0]
        t0, t1 = r * ROW_SEC, (r + 1) * ROW_SEC
        ticks = np.arange(t0, t1 + 1, 10)
        for p in result['idle']:
            ax.axvspan(p['start'], p['end'], color='tab:green' if p['idle'] else 'tab:orange', alpha=0.25, lw=0)
        for e in result['episodes']:
            ax.axvspan(e['start'], e['end'], color='tab:blue', alpha=0.08, lw=0)
        for name, values in zip(names, d['gyro_rms']):
            ax.plot(grid, values, lw=0.6, label=name)
        ax.axhline(params['gyro_still'], color='k', lw=0.6, ls='--')
        docked = d['frame_times'][d['frame_docked']]
        ax.plot(docked, np.full(len(docked), 5.0), '|', color='k', ms=4)
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 10)
        ax.set_ylabel('gyro RMS rad/s', fontsize=7)
        for name, times, high in zip(names, d['accel_times'], d['accel_high']):
            inside = (times >= t0) & (times <= t1)
            ax_accel.plot(times[inside], high[inside], lw=0.5, label=name)
        ax_accel.axhline(params['bang_accel'], color='k', lw=0.6, ls='--')
        ax_accel.set_ylabel('accel spike m/s^2', fontsize=7)
        inside = (d['audio_times'] >= t0) & (d['audio_times'] <= t1)
        ax_audio.plot(d['audio_times'][inside], d['audio_ratio'][inside], lw=0.5, color='tab:purple')
        ax_audio.axhline(params['audio_ratio'], color='k', lw=0.6, ls='--')
        ax_audio.set_ylabel('head audio click', fontsize=7)
        for a in (ax, ax_accel, ax_audio):
            for t in result['bangs']:
                a.axvline(t, color='tab:red', lw=1.0)
            a.set_xlim(t0, t1)
            a.set_xticks(ticks)
            a.set_xticklabels([mmss(t) for t in ticks], fontsize=7)
    axes[0, 0].legend(loc='upper right', fontsize=7)
    axes[0, 0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def print_details(result, params):
    d = result['diagnostics']
    joint = d['joint']
    bangs = np.array(result['bangs'])
    docked_d = d['frame_dock_distance'][d['frame_docked']]
    print(f"  head frames with docked view: {d['frame_docked'].sum()} ({d['frame_docked'].mean() * 100:.1f}%)"
        + (f", cube to dock distance median {np.median(docked_d):.3f} m, max {np.max(docked_d):.3f} m" if len(docked_d) else ''))
    print(f"  head audio clicks >= {params['audio_ratio']}: {result['n_clicks']}, "
        f"click ratio percentiles 50/99/99.9: {np.round(np.percentile(d['audio_ratio'], [50, 99, 99.9]), 1).tolist()}")
    print(f"  joint spikes in the first / last {REPORT_WINDOW_SEC:.0f}s of each stretch: "
        f"time strength/dt/audio[/dock], * = bang")
    for e in d['stretches']:
        print(f"    {mmss(e['active_start'])}-{mmss(e['active_end'])} {e['status']}"
            + (f" episode {mmss(e['start'])}-{mmss(e['end'])}" if e['status'] == 'ok' else ''))
        for label, a, b in [('start', e['active_start'], e['active_start'] + REPORT_WINDOW_SEC),
                ('end', e['active_end'] - REPORT_WINDOW_SEC, e['active_end'])]:
            near = np.flatnonzero((joint['time'] >= a) & (joint['time'] <= b))
            items = [f"{mmss(joint['time'][k])} {joint['strength'][k]:.0f}/{joint['dt'][k] * 1000:+.0f}ms/"
                f"{joint['audio'][k]:.1f}" + ('/dock' if joint['docked'][k] else '')
                + ('*' if len(bangs) and np.min(np.abs(bangs - joint['time'][k])) < 1e-6 else '')
                for k in near]
            print(f"      {label}: " + (', '.join(items) if items else '-'))


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml (episodes section)')
def main(input, config):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'episodes' in config, "tag tracking config needs an episodes section"
    params = {k: v for k, v in config['episodes'].items() if k not in ('dock_tags', 'max_gap_sec')}
    inputs = segment_step.take_inputs(config)
    output_dir = session.joinpath('tag_tracking', 'wrist_events')
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = get_session_videos(session)

    for head in [x for x in videos if x['camera_serial'] == config['head_camera']]:
        gripper_wrists = [overlapping(videos, head, serial) for serial in inputs['serials']]
        result = segment_take(head, gripper_wrists, inputs['cube_tags'], inputs['dock_tags'],
            config['tag_sizes'], config['cameras'][config['head_camera']], params)
        segment_step.print_summary(head['name'], result)
        for name, values in zip(inputs['names'], result['diagnostics']['gyro_rms']):
            print(f"  {name}: gyro RMS percentiles 5/25/50/75 "
                f"{np.round(np.nanpercentile(values, [5, 25, 50, 75]), 4).tolist()} rad/s")
        print_details(result, params)
        path = output_dir.joinpath(head['name'] + '.png')
        plot_take(str(path), head['name'], inputs['names'], params, result)
        print(f"  saved {path}")

# %%
if __name__ == "__main__":
    main()
