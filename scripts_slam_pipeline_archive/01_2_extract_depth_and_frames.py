import sys
import os
import pathlib
import shutil
import datetime
import click
import numpy as np
import concurrent.futures
from tqdm import tqdm
import multiprocessing

# ROS/CV 의존성 (실제 ROS 환경에 맞게 설치 및 구성 필수)
try:
    import rosbag
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge  # 이미지 변환에 필요
    import cv2  # 이미지 저장 및 변환에 필요
except ImportError:
    print(
        "FATAL ERROR: ROS libraries (rosbag, cv_bridge, cv2) must be installed and sourced."
    )
    sys.exit(1)


# --- 상수 정의 ---
DEFAULT_COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEFAULT_DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"

# CvBridge 인스턴스 (전역에서 한 번 생성)
try:
    bridge = CvBridge()
except:
    bridge = None

# ----------------------------------------------------------------------
## 헬퍼 함수
# ----------------------------------------------------------------------


def bag_get_start_datetime(bag_path):
    # Dummy implementation using file modification time for simplicity
    mtime = pathlib.Path(bag_path).stat().st_mtime
    return datetime.datetime.fromtimestamp(mtime)


def extract_realsense_frames_and_depth(
    bag_path: pathlib.Path,
    bag_dir: pathlib.Path,
    color_topic: str = DEFAULT_COLOR_TOPIC,
    depth_topic: str = DEFAULT_DEPTH_TOPIC,
):
    """
    Extracts Color frames (PNG) and Depth frames (RAW binary) 
    from the BAG file and creates TUM-style timestamp files.
    """
    if bridge is None:
        print(f"Skipping {bag_path.name}: CvBridge not initialized.")
        return -1

    output_dir = bag_dir.joinpath("frames")
    color_dir = output_dir.joinpath("color")
    depth_dir = output_dir.joinpath("depth")

    # Clean and create output directories
    if output_dir.exists():
        shutil.rmtree(output_dir)
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_timestamps = []
    depth_timestamps = []
    
    topics_to_read = [color_topic, depth_topic]
    
    try:
        with rosbag.Bag(str(bag_path), "r") as bag:
            for topic, msg, t in bag.read_messages(topics=topics_to_read):
                if msg._type != Image._type:
                    continue
                
                timestamp_sec = msg.header.stamp.to_sec()
                timestamp_str = f"{timestamp_sec:.6f}"
                
                if topic == color_topic:
                    # Color Frame: Save as PNG and record timestamp
                    # Use BGR8 for standard OpenCV PNG writing
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    filename = f"{timestamp_str}.png"
                    cv2.imwrite(str(color_dir.joinpath(filename)), cv_image)
                    rgb_timestamps.append(f"{timestamp_str} color/{filename}")
                    
                elif topic == depth_topic:
                    # Depth Frame: Save as RAW binary data (CV_16U format)
                    cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    
                    # 1. Save RAW depth data for SLAM processing
                    # Filename format for TUM standard
                    raw_filename = f"{timestamp_str}.raw"
                    raw_filepath = depth_dir.joinpath(raw_filename)
                    
                    # Convert to bytes and write to file
                    cv_depth.tofile(str(raw_filepath)) 
                    
                    # 2. Record timestamp in TUM format (depth.txt)
                    depth_timestamps.append(f"{timestamp_str} depth/{raw_filename}")

        # Save TUM-style timestamp lists
        with output_dir.joinpath("rgb.txt").open("w") as f:
            f.write("# timestamp filename\n")
            f.write("\n".join(rgb_timestamps))
            
        with output_dir.joinpath("depth.txt").open("w") as f:
            f.write("# timestamp filename\n")
            f.write("\n".join(depth_timestamps))
            
        total_extracted = len(rgb_timestamps) + len(depth_timestamps)
        return total_extracted
        
    except Exception as e:
        print(f"Error processing {bag_path.name} (Depth/Frames): {e}")
        # Clean up incomplete extraction attempt
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return -1


# ----------------------------------------------------------------------
## 메인 CLI 함수
# ----------------------------------------------------------------------


@click.command(help="Extracts Color/Depth frames and timestamp files from raw_bag.bag.")
@click.option(
    "-c", "--color_topic", default=DEFAULT_COLOR_TOPIC, help="RealSense Color topic."
)
@click.option(
    "-d", "--depth_topic", default=DEFAULT_DEPTH_TOPIC, help="RealSense Depth topic."
)
@click.option(
    "-n",
    "--num_workers",
    type=int,
    default=None,
    help="Number of concurrent processes.",
)
@click.argument("session_dir", nargs=-1)
def main(color_topic, depth_topic, num_workers, session_dir):
    # Set up multiprocessing environment
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # --- 환경 경로 설정 ---
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    print("--- Starting RealSense BAG Frames/Depth Extraction ---")

    for session in session_dir:
        session_path = pathlib.Path(os.path.expanduser(session)).absolute()
        input_bag_paths = [x for x in session_path.glob("**/raw_bag.bag")]

        if not input_bag_paths:
            print(
                f"Warning: No 'raw_bag.bag' found in directories under {session_path}. Skipping."
            )
            continue

        print(f"Found {len(input_bag_paths)} BAG files for frame/depth extraction.")

        total_tasks = len(input_bag_paths)
        
        with tqdm(total=total_tasks) as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = set()

                for bag_path in tqdm(
                    input_bag_paths, desc="Scheduling frame/depth extraction"
                ):
                    bag_dir = bag_path.parent
                    
                    # 1. Depth and Frame extraction task
                    frame_future = executor.submit(
                        extract_realsense_frames_and_depth,
                        bag_path,
                        bag_dir,
                        color_topic,
                        depth_topic,
                    )
                    futures.add(frame_future)

                    if len(futures) >= num_workers:
                        completed, futures = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        pbar.update(len(completed))

                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))

        # Result summary
        results = [
            x.result() for x in completed if x.result() != 0 and x.result() != -1
        ]
        errors = [x.result() for x in completed if x.result() == -1]

        print("\nDone! Summary:")
        print(f"  Total successful frame/depth extractions: {len(results)}")
        print(f"  Files with errors: {len(errors)}")


# %%
if __name__ == "__main__":
    # Simulate setup to allow standalone execution testing
    # ROOT_DIR determination must be outside main if used globally
    if len(sys.argv) == 1:
        main.main(["--help"])
    else:
        main()