"""
python scripts_slam_pipeline/01_extract_gopro_imu.py data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm

# %%
@click.command()
@click.option('-d', '--docker_image', default="chicheng/openicc:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.argument('session_dir', nargs=-1)
def main(docker_image, num_workers, no_docker_pull, session_dir):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    for session in session_dir:
        input_dir = pathlib.Path(os.path.expanduser(session)).joinpath('demos')
        input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
        print(f'Found {len(input_video_dirs)} video dirs')

        with tqdm(total=len(input_video_dirs)) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                for video_dir in tqdm(input_video_dirs):
                    video_dir = video_dir.absolute()
                    if video_dir.joinpath('imu_data.json').is_file():
                        print(f"imu_data.json already exists, skipping {video_dir.name}")
                        continue
                    mount_target = pathlib.Path('/data')

                    video_path = mount_target.joinpath('raw_video.mp4')
                    json_path = mount_target.joinpath('imu_data.json')

                    # run imu extractor
                    cmd = [
                        'docker',
                        'run',
                        '--rm', # delete after finish
                        '--volume', str(video_dir) + ':' + '/data',
                        docker_image,
                        'node',
                        '/OpenImuCameraCalibrator/javascript/extract_metadata_single.js',
                        str(video_path),
                        str(json_path)
                    ]

                    stdout_path = video_dir.joinpath('extract_gopro_imu_stdout.txt')
                    stderr_path = video_dir.joinpath('extract_gopro_imu_stderr.txt')

                    if len(futures) >= num_workers:
                        # limit number of inflight tasks
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))

                    futures.add(executor.submit(
                        lambda x, stdo, stde: subprocess.run(x, 
                            cwd=str(video_dir),
                            stdout=stdo.open('w'),
                            stderr=stde.open('w')), 
                        cmd, stdout_path, stderr_path))
                    # print(' '.join(cmd))

                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))

        print("Done! Result:")
        print([x.result() for x in completed])

# %%
if __name__ == "__main__":
    main()
