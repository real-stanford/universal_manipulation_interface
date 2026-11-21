"""
scripts_tag_tracking/process_station.py cleanup: deletes the local videos and MCAP files
only when every MCAP file is on S3 with the same size; keeps everything else.

python tests/test_process_station.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))

import pathlib
import tempfile
import process_station


def make_session(root: pathlib.Path) -> pathlib.Path:
    s = root.joinpath('task01')
    for d in ['raw_videos/gripper_calibration', 'demos/demo_A', 'demos/gripper_calibration_A', 'dataset_mcap',
            'tag_tracking', 'logs']:
        s.joinpath(d).mkdir(parents=True)
    s.joinpath('demos/demo_A/raw_video.mp4').write_bytes(b'v' * 100)
    s.joinpath('demos/demo_A/tag_corners.pkl').write_bytes(b'x')
    s.joinpath('demos/gripper_calibration_A/raw_video.mp4').write_bytes(b'v' * 50)
    s.joinpath('raw_videos/GX01.MP4').symlink_to('../demos/demo_A/raw_video.mp4')
    s.joinpath('raw_videos/gripper_calibration/GX02.MP4').symlink_to('../../demos/gripper_calibration_A/raw_video.mp4')
    s.joinpath('raw_videos/pairs.csv').write_text('type\n')
    s.joinpath('dataset_mcap/episode_000000.mcap').write_bytes(b'm' * 10)
    s.joinpath('dataset_mcap/episode_000001.mcap').write_bytes(b'm' * 20)
    s.joinpath('dataset_plan.pkl').write_bytes(b'p')
    return s


def test_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        s = make_session(pathlib.Path(tmp))
        # one MCAP file missing on S3: nothing deleted
        process_station.s3_sizes = lambda url: {'episode_000000.mcap': 10}
        assert 'skipped' in process_station.cleanup(s, 's3://b/x/task01')
        # a size differs: nothing deleted
        process_station.s3_sizes = lambda url: {'episode_000000.mcap': 10, 'episode_000001.mcap': 19}
        assert 'skipped' in process_station.cleanup(s, 's3://b/x/task01')
        assert s.joinpath('demos/demo_A/raw_video.mp4').is_file() and len(list(s.rglob('*.mcap'))) == 2
        # all there: videos, raw_videos entries and MCAP files deleted, the rest kept
        process_station.s3_sizes = lambda url: {'episode_000000.mcap': 10, 'episode_000001.mcap': 20, 'other': 1}
        done = process_station.cleanup(s, 's3://b/x/task01')
        assert 'deleted 2 videos, 2 raw_videos entries, 2 MCAP files' in done, done
        assert not list(s.rglob('*.mp4')) and not list(s.rglob('*.MP4')) and not list(s.rglob('*.mcap'))
        for kept in ['raw_videos/pairs.csv', 'demos/demo_A/tag_corners.pkl', 'dataset_plan.pkl']:
            assert s.joinpath(kept).is_file(), kept
        # nothing local any more: skipped, not an error
        assert 'skipped' in process_station.cleanup(s, 's3://b/x/task01')


if __name__ == '__main__':
    test_cleanup()
    print('all passed')
