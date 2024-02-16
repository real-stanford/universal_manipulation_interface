# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import json
import pathlib
import numpy as np
from tqdm import tqdm
from exiftool import ExifToolHelper

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    with ExifToolHelper() as et:
        for session in tqdm(session_dir):
            session = pathlib.Path(os.path.expanduser(session)).absolute()
            # hardcode subdirs
            demos_dir = session.joinpath('demos')
            
            mp4_paths = list(demos_dir.glob("demo_*/raw_video.mp4"))
            if len(mp4_paths) < 1:
                continue
            all_meta = et.get_tags(mp4_paths, ['QuickTime:AutoRotation'])
            for mp4_path, meta in zip(mp4_paths, all_meta):
                rot = meta['QuickTime:AutoRotation']
                if rot != 'U':
                    demo_dir = mp4_path.parent
                    print(f"Found rotated video: {session.name} {demo_dir.name}")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
