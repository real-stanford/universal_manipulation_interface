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


def test_station_job():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp, 'station3', 'industrial_data')
        for t in ['task01', 'task02']:
            root.joinpath(t, 'raw_videos').mkdir(parents=True)
        root.joinpath('notes').mkdir()
        config = pathlib.Path(tmp, 'config.yaml')
        config.write_text('setup_calibration: setup_x\n')
        job = process_station.station_job(str(root), str(config), None, None, True, None, False)
        assert job['name'] == 'station3' and job['names'] == ['task01', 'task02'] and not job['on_s3']
        assert process_station.station_job(str(root), str(config), None, ['task02'], True, None, False)['names'] == ['task02']
        for bad in [dict(tasks='task03'), dict(cleanup_after=True), dict(upload='s3://b/x', mcap=False)]:
            args = dict(tasks=None, mcap=True, upload=None, cleanup_after=False)
            args.update(bad)
            try:
                process_station.station_job(str(root), str(config), None, args['tasks'], args['mcap'], args['upload'],
                    args['cleanup_after'])
                raise RuntimeError(f"accepted {bad}")
            except AssertionError:
                pass


def test_batched_mcap():
    import pickle
    with tempfile.TemporaryDirectory() as tmp:
        s = make_session(pathlib.Path(tmp))
        pickle.dump([{}] * 23, s.joinpath('dataset_plan.pkl').open('wb'))
        uploaded, exports = dict(), list()

        def fake_run(cmd, log):
            cmd = [str(c) for c in cmd]
            if '07_export_mcap.py' in ' '.join(cmd):
                batch = [int(k) for k in cmd[cmd.index('-e') + 1].split(',')]
                exports.append(batch)
                assert len(list(s.joinpath('dataset_mcap').glob('*.mcap'))) == 0, "earlier batch not deleted"
                for k in batch:
                    s.joinpath('dataset_mcap', f'episode_{k:06d}.mcap').write_bytes(b'm' * (k + 1))
            elif cmd[:3] == ['aws', 's3', 'sync']:
                for p in s.joinpath('dataset_mcap').glob('*.mcap'):
                    uploaded[p.name] = p.stat().st_size
            return 0, []
        process_station.run = fake_run
        process_station.s3_sizes = lambda url: dict(uploaded)
        job = {'name': 'station1', 'sessions': tmp, 'config': 'c.yaml', 'on_s3': False, 'root': s.parent,
            'names': ['task01'], 'upload': 's3://b/x'}
        summary = process_station.process(job, False, True, True, None, start_from='mcap', batch_size=10)
        assert exports == [list(range(10)), list(range(10, 20)), list(range(20, 23))], exports
        name, status, failed, plan_yield, _ = summary[0]
        assert status == 'ok', failed
        assert len(uploaded) == 23 and '23 MCAP files' in plan_yield and 'cleanup: deleted 2 videos' in plan_yield, plan_yield
        assert not list(s.rglob('*.mcap')) and not list(s.rglob('*.mp4'))
        # an upload that does not arrive: the batch stays local, the session fails, videos kept
        s2 = make_session(pathlib.Path(tmp, 'other'))
        pickle.dump([{}] * 5, s2.joinpath('dataset_plan.pkl').open('wb'))
        s = s2
        uploaded.clear()
        process_station.s3_sizes = lambda url: {}
        job.update({'root': s2.parent})
        summary = process_station.process(job, False, True, True, None, start_from='mcap', batch_size=10)
        assert summary[0][1] == 'failed' and 'not on S3' in summary[0][2], summary
        assert len(list(s2.rglob('*.mcap'))) == 5 and s2.joinpath('demos/demo_A/raw_video.mp4').is_file()


def test_download_space_check():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp, 'sessions')
        calls = list()
        process_station.run = lambda cmd, log: calls.append(cmd) or (0, [])
        process_station.s3_total_bytes = lambda url: 10 ** 15   # 1 PB: never fits
        job = {'name': 'station3', 'sessions': 's3://b/station3/industrial_data', 'config': 'c.yaml', 'on_s3': True,
            'root': root, 'names': ['task01', 'task02'], 'upload': 's3://b/up'}
        summary = process_station.process(job, False, True, True, None)
        assert [x[1] for x in summary] == ['skipped', 'skipped'] and 'enlarge the disk' in summary[0][2], summary
        assert calls == [], "nothing may run (no download) when the session does not fit"


if __name__ == '__main__':
    test_download_space_check()
    test_batched_mcap()
    test_station_job()
    test_cleanup()
    print('all passed')
