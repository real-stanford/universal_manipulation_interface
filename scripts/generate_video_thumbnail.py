# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import av
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import cv2
import numpy as np

# %%
def worker(in_path, out_path, out_res, down_sample_ratio, speed_up):
    with av.open(str(out_path), mode='w') as out_container:
        with av.open(str(in_path)) as in_container:
            in_stream = in_container.streams.video[0]
            in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            
            out_rate = int(float(in_stream.average_rate) / down_sample_ratio * speed_up)
            out_stream = out_container.add_stream('h264', rate=out_rate)
            out_stream.thread_type = 'AUTO'
            out_stream.thread_count = 1
            
            rw, rh = out_res

            out_stream.width = rw
            out_stream.height = rh
            
            out_codec_context = out_stream.codec_context
            out_codec_context.options = {
                'crf': '21',
                'profile': 'high'
            }
            
            for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
                if i % int(down_sample_ratio) != 0:
                    continue
                
                img = frame.to_ndarray(format='rgb24')
                img = cv2.resize(img, (rh, rw), interpolation=cv2.INTER_AREA)
                
                out_frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)
            
            # flush
            for packet in out_stream.encode():
                out_container.mux(packet)


# %%
@click.command()
@click.argument('videos', nargs=-1)
@click.option('-or', '--out_res', type=str, default='400x300')
@click.option('-ds', '--down_sample_ratio', type=int, default=8)
@click.option('-s', '--speed_up', type=float, default=2.0)
@click.option('-n', '--num_workers', type=int, default=None)
def main(videos, num_workers, out_res, down_sample_ratio, speed_up):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
    
    out_res = tuple(int(x) for x in out_res.split('x'))
    
    vid_args = list()
    for video in videos:
        video_path = pathlib.Path(os.path.expanduser(video))
        out_path = video_path.parent.joinpath(video_path.stem + '.th' + video_path.suffix)
        vid_args.append((video_path, out_path))
    
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for args in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(
                    worker, *args, 
                    out_res=out_res, 
                    down_sample_ratio=down_sample_ratio,
                    speed_up=speed_up))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])
            
# %%
if __name__ == "__main__":
    main()
