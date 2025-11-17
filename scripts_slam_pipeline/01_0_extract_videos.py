import sys
import os
import pathlib
import click
import multiprocessing
import concurrent.futures
from tqdm import tqdm

# ROS and CV dependencies (MUST be installed and configured in the environment)
try:
    import rosbag
    from cv_bridge import CvBridge
    import cv2
    from sensor_msgs.msg import Image
except ImportError:
    print("FATAL ERROR: ROS libraries (rosbag, cv_bridge, cv2) must be installed and sourced.")
    sys.exit(1)

# --- Environment Setup (Boilerplate) ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# --- Helper Function: BAG to MP4 Conversion ---

def process_bag_to_mp4(bag_path, mp4_path, color_topic_name='/device_0/sensor_1/Color_0/image/data', fps=10):
    """
    Core conversion function: Reads the color image topic from a BAG file and saves it as an MP4.
    
    Args:
        bag_path (pathlib.Path): Path to the source raw_bag.bag file.
        mp4_path (pathlib.Path): Path to the target raw_video.mp4 file.
        color_topic_name (str): ROS topic name for the color image stream.
        fps (int): Target frame rate for the output MP4 video.
    
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    print(f"[MP4] Converting {bag_path.parent.name} to MP4...")
    
    bridge = CvBridge()
    video_writer = None
    success = False
    
    try:
        with rosbag.Bag(str(bag_path), 'r') as bag:
            is_first_frame = True
            
            # Read BAG file and write frames to video
            for topic_loop, msg_loop, t_loop in bag.read_messages(topics=[color_topic_name]):
                # Check for correct message type
                if msg_loop._type != Image._type: 
                     continue
                
                # Convert image message to OpenCV Mat object
                cv_image = bridge.imgmsg_to_cv2(msg_loop, desired_encoding="bgr8")
                
                if is_first_frame:
                    height, width, layers = cv_image.shape
                    
                    # VideoWriter setup (Using 'mp4v' codec for broad compatibility)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    # Use absolute path for robustness
                    video_writer = cv2.VideoWriter(str(mp4_path.absolute()), fourcc, fps, (width, height))
                    is_first_frame = False
                    print(f"[MP4] Writer initialized for {bag_path.parent.name}: {width}x{height} @ {fps} FPS")

                if video_writer is not None:
                    video_writer.write(cv_image)
            
            success = video_writer is not None and not is_first_frame
            
    except Exception as e:
        print(f"[MP4] Conversion failed for {bag_path.parent.name} due to an exception: {e}")
        success = False
    finally:
        if video_writer is not None:
            video_writer.release()
            
    if success:
        print(f"[MP4] Successfully converted {bag_path.parent.name} to {mp4_path.name}")
    else:
        print(f"[MP4] Conversion failed for {bag_path.parent.name}: No image frames found or initialization failed.")
        
    return success

# --- Main CLI Function ---

@click.command(help='Extracts MP4 video from raw_bag.bag files in demos subdirectories.')
@click.option('-c', '--color_topic', default='/device_0/sensor_1/Color_0/image/data', help="RealSense Color topic.")
@click.option('-f', '--fps', type=int, default=30, help="Target FPS for the output MP4 video.")
@click.option('-n', '--num_workers', type=int, default=None, help="Number of concurrent processes.")
@click.argument('session_dir', nargs=-1)
def main(color_topic, fps, num_workers, session_dir):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        print(f"Using {num_workers} workers.")

    print("--- Starting RealSense BAG to MP4 Video Conversion ---")

    for session in session_dir:
        session_path = pathlib.Path(os.path.expanduser(session)).absolute()
        input_dir = session_path.joinpath('demos')
        
        # Find all raw_bag.bag paths
        input_bag_paths = [x for x in input_dir.glob('*/raw_bag.bag')]
        
        if not input_bag_paths:
            print(f"Warning: No 'raw_bag.bag' found in directories under {input_dir}. Skipping session.")
            continue

        print(f'Found {len(input_bag_paths)} BAG files for MP4 conversion.')

        total_tasks = len(input_bag_paths)

        # ProcessPoolExecutor를 사용하여 병렬 처리 (CV 작업은 CPU 바인딩이므로 프로세스 풀 사용)
        with tqdm(total=total_tasks) as pbar: 
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                
                for bag_path in tqdm(input_bag_paths, desc="Scheduling MP4 conversion"):
                    bag_dir = bag_path.parent 
                    mp4_path = bag_dir.joinpath('raw_video.mp4')

                    # Skip if MP4 already exists
                    if mp4_path.is_file():
                        print(f"[INFO] {bag_dir.name}/raw_video.mp4 already exists. Skipping.")
                        pbar.update(1)
                        continue

                    # MP4 추출 작업 예약
                    mp4_future = executor.submit(
                        process_bag_to_mp4, bag_path, mp4_path, color_topic, fps)
                    futures.add(mp4_future)
                    
                    # 완료된 작업 처리 및 tqdm 업데이트
                    if len(futures) >= num_workers:
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))

                # 남아있는 모든 작업 완료 대기
                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))

        # 결과 요약
        results = [x.result() for x in completed if x.result() is True]
        errors = [x.result() for x in completed if x.result() is False]
        
        print("\nDone! Summary:")
        print(f"  Total successful MP4 conversions: {len(results)}")
        print(f"  Total conversion failures: {len(errors)}")

# %%
if __name__ == "__main__":
    main()