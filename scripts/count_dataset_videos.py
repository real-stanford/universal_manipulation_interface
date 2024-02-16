# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import pickle
import collections
import click

# %%
@click.command()
@click.argument('input', nargs=-1)
def main(input):
    n_videos = 0
    n_episode = 0
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        
        # generate argumnet for videos
        n_grippers = None
        n_cameras = None
        buffer_start = 0
        videos_dict = collections.defaultdict(list)
        for plan_episode in plan:
            n_episode += 1
            grippers = plan_episode['grippers']
            
            # check that all episodes have the same number of grippers 
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
            
            # aggregate video gen aguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()
                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames
        
        # vid_args.extend(videos_dict.items())
        n_videos += len(videos_dict)
    print(f"{n_videos} videos and {n_episode} episodes in total.")

# %%
if __name__ == "__main__":
    main()
