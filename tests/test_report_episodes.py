"""
scripts_tag_tracking/report_episodes.py: the ranged-read file view gives the MCAP reader the
same metadata as reading the file directly (S3 fetch replaced by local reads), and
episode_info counts an episode with missing pose steps as not good.

python tests/test_report_episodes.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))

import tempfile
from mcap.writer import Writer
import report_episodes


def write_mcap(path, missing, steps=600):
    with open(path, 'wb') as f:
        w = Writer(f)
        w.start()
        schema = w.register_schema('x', 'jsonschema', b'{}')
        channel = w.register_channel('robot0/gripper/state', 'json', schema)
        for k in range(steps):
            w.add_message(channel, log_time=k * 16_683_350, publish_time=k * 16_683_350, data=b'{"width": 0.05}' * 20)
        w.add_metadata('recording', {'episode_length': str(steps), 'video_fps': '59.94005994005994', 'n_grippers': '2',
            'robot0_pose_missing_steps': str(missing[0]), 'robot1_pose_missing_steps': str(missing[1])})
        w.finish()


def test_report():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'station1', 'task01'))
        good, gaps = os.path.join(tmp, 'station1', 'task01', 'episode_000000.mcap'), os.path.join(tmp, 'station1', 'task01', 'episode_000001.mcap')
        write_mcap(good, (0, 0))
        write_mcap(gaps, (0, 12))
        a = report_episodes.episode_info(good, os.path.getsize(good))
        assert a['good'] and abs(a['duration_s'] - 600 / 59.94005994005994) < 1e-9, a
        assert not report_episodes.episode_info(gaps, os.path.getsize(gaps))['good']
        # the S3 view, blocks from the local file, small blocks so reads cross block borders
        data = open(gaps, 'rb').read()
        report_episodes.S3File.BLOCK = 4096
        report_episodes.S3File.block = lambda self, k: data[k * self.BLOCK:(k + 1) * self.BLOCK]
        b = report_episodes.episode_info('s3://bucket/x/episode_000001.mcap', len(data))
        assert b == report_episodes.episode_info(gaps, len(data)), b
        files = report_episodes.list_mcaps(tmp)
        assert [f[2] for f in files] == ['station1/task01', 'station1/task01']


if __name__ == '__main__':
    test_report()
    print('all passed')
