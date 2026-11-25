"""
Segment each continuous head camera take into episodes, before tag tracking
(umi.tag_tracking.episodes.segment_take, rules in umi.tag_tracking.wrist_events):
docked (both wrist IMUs still, head sees both cubes at the dock tags)
-> bang -> task -> bang -> docked. A bang: spikes in both wrist IMUs at once,
with the click in the head camera audio, not at the dock.
Diagnostic: per wrist video, its audio clock against the head audio from the
bang clicks (umi.tag_tracking.wrist_events.click_offset, wrist_clock_line);
not applied (GoPro audio is not locked to the video), the wrist clocks are
corrected from motion by 03s_sync_wrists.py.
Settings: the 'episodes' section of the tag tracking config.

Requires: wrist demos/*/imu_data.json (01_extract_gopro_imu.py),
audio_sync_offsets.json (06a_estimate_audio_sync.py),
head demos/*/tag_corners.pkl (00_detect_tag_corners.py).
Writes <session>/tag_tracking/episodes.json, times in seconds of each head video:
{
  "parameters": {...},
  "videos": {
    "<head video dir>": {
      "episodes": [{"start", "end", "active_start", "active_end", "n_bangs",
                    "audio_clock": {"<wrist video dir>": {"gripper", "offset_ms", "residual_ms",
                                    "bang_offsets_ms"} or null}}, ...],
      "rejected": [{"active_start", "active_end", "status", "n_bangs"}, ...],
      "idle": [{"start", "end", "n_docked_frames", "idle"}, ...],
      "bangs": [...], "bang_strengths": [...], "bang_wrist_dt_ms": [...],
      "audio_clock": {"<wrist video dir>": {"gripper", "offset_ms", "drift_ms_per_min",
                     "n_bangs", "n_used", "n_inliers", "residual_ms", "bangs": [...]} or null},
      "audio_lag", "n_lag_pairs", "n_clicks"
    }
  }
}
Plots and per-bang details: scripts_tag_tracking/check_wrist_events.py.

python scripts_tag_tracking/00b_segment_episodes.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import numpy as np
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.episodes import segment_take


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def take_inputs(config):
    """
    Gripper names (robot order), cube tags, wrist serials, dock tags.
    """
    names = [name for _, name in sorted((b['robot_index'], name)
        for name, b in config['bodies'].items() if 'gripper_camera' in b)]
    dock_tags = config['episodes']['dock_tags']
    return {
        'names': names,
        'cube_tags': [config['bodies'][name]['tags'] for name in names],
        'serials': [config['bodies'][name]['gripper_camera'] for name in names],
        'dock_tags': dock_tags
    }


def print_summary(name, result):
    episodes = result['episodes']
    durations = [e['end'] - e['start'] for e in episodes]
    lag = result['audio_lag']
    print(f"{name}: {len(episodes)} episodes"
        + (f" ({min(durations):.1f}-{max(durations):.1f}s, total {sum(durations):.0f}s)" if durations else '')
        + f", {sum(p['idle'] for p in result['idle'])} idle periods, {len(result['bangs'])} bangs, "
        f"{len(result['rejected'])} rejected stretches; wrist IMU to head audio lag "
        + (f"{lag * 1000:.0f} ms {result['audio_lag_drift_ms_per_min']:+.2f} ms/min ({result['n_lag_pairs']} pairs)"
            if lag is not None else "not estimated")
        + f"; wrist-to-wrist {result['wrist_dt_ms'][0]:+.1f} ms {result['wrist_dt_ms'][1]:+.2f} ms/min")
    for wrist_name, line in result.get('audio_clock', dict()).items():
        print(f"  audio clock (diagnostic) {wrist_name}: " + ('no clock line (fewer than 2 bangs matched)' if line is None else
            f"{line['offset_ms']:+.1f} ms {line['drift_ms_per_min']:+.2f} ms/min from {line['n_inliers']} of "
            f"{line['n_bangs']} bangs ({line['n_bangs'] - line['n_used']} correlated below the threshold, "
            f"{line['n_used'] - line['n_inliers']} outliers), residual RMS {line['residual_ms']:.2f} ms"
            + ('' if 'imu_to_audio' not in line else "; own IMU spike to own audio click {:+.1f} ms {:+.2f} ms/min "
                "(residual RMS {:.1f} ms, {} bangs)".format(line['imu_to_audio']['offset_ms'],
                line['imu_to_audio']['drift_ms_per_min'], line['imu_to_audio']['residual_ms'],
                line['imu_to_audio']['n_inliers']))))
    bangs = np.array(result['bangs'])
    raw_dt, corrected_dt = list(), list()
    for k, e in enumerate(episodes):
        sync = e.get('audio_clock', dict())
        items = [name.split('_')[1][-6:] + ' ' + ('none' if x is None else '{:+.1f} ms (bangs {})'.format(
            x['offset_ms'], ' '.join('{:+.1f}'.format(v) for v in x['bang_offsets_ms']) or '-'))
            for name, x in sync.items()]
        print(f"  episode {k + 1}: {mmss(e['start'])}-{mmss(e['end'])} ({e['end'] - e['start']:.1f}s, {e['n_bangs']} bangs)"
            + (f"; audio clock {', '.join(items)}" if items else ''))
        # cross-check: wrist-to-wrist IMU spike time difference before / after the correction
        offsets = {x['gripper']: x['offset_ms'] for x in sync.values() if x is not None}
        if len(offsets) == 2:
            for j in np.flatnonzero((bangs >= e['start']) & (bangs <= e['end'])):
                raw_dt.append(result['bang_wrist_dt_ms'][j])
                corrected_dt.append(result['bang_wrist_dt_ms'][j] + offsets[1] - offsets[0])
    if raw_dt:
        print(f"  wrist-to-wrist IMU spike difference at the bangs, {len(raw_dt)} bangs: "
            f"audio sync only {np.min(raw_dt):+.1f} to {np.max(raw_dt):+.1f} ms (std {np.std(raw_dt):.1f}), "
            f"with the audio clock lines {np.min(corrected_dt):+.1f} to {np.max(corrected_dt):+.1f} ms "
            f"(std {np.std(corrected_dt):.1f}); a constant remainder is the cameras' fixed IMU timing difference")
    for r in result['rejected']:
        print(f"  rejected {mmss(r['active_start'])}-{mmss(r['active_end'])}: {r['status']}")
    for p in result['idle']:
        if not p['idle']:
            print(f"  both still, head did not see the dock: {mmss(p['start'])}-{mmss(p['end'])}")


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
def main(input, config):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'episodes' in config, "tag tracking config needs an episodes section"
    params = {k: v for k, v in config['episodes'].items() if k not in ('dock_tags', 'max_gap_sec', 'dock_exit_distance', 'fixed_trim_sec')}
    inputs = take_inputs(config)
    videos = get_session_videos(session)

    results = dict()
    for head in [x for x in videos if x['camera_serial'] == config['head_camera']]:
        gripper_wrists = [overlapping(videos, head, serial) for serial in inputs['serials']]
        if any(len(w) == 0 for w in gripper_wrists):
            # e.g. a short false start recorded by the head camera only: no episodes to segment
            print(f"WARNING skipped {head['name']}: no wrist video of every gripper overlaps it")
            continue
        result = segment_take(head, gripper_wrists, inputs['cube_tags'], inputs['dock_tags'],
            config['tag_sizes'], config['cameras'][config['head_camera']], params)
        result.pop('diagnostics')
        results[head['name']] = result
        print_summary(head['name'], result)

    output = session.joinpath('tag_tracking', 'episodes.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'parameters': config['episodes'], 'videos': results}, output.open('w'), indent=2)
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
