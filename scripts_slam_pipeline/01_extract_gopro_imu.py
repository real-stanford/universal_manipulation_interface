#!/usr/bin/env python3
# %%
import sys
import os
import pathlib
import click
import multiprocessing
import concurrent.futures
import json
from tqdm import tqdm
import datetime
import shutil # shutil은 main 함수에서 사용되므로 import 유지

# ROS/IMU 의존성 (실제 ROS 환경에 맞게 설치 및 구성 필수)
try:
    import rosbag
    import rospy
    from sensor_msgs.msg import Imu
    from diagnostic_msgs.msg import KeyValue # KeyValue는 datetime 헬퍼에서 필요
    from cv_bridge import CvBridge # 다른 헬퍼 함수에서 필요
    import cv2 # 다른 헬퍼 함수에서 필요
    from sensor_msgs.msg import Image # 다른 헬퍼 함수에서 필요
    # rosbag get_start_time/serial/process_bag_to_mp4 헬퍼 함수는 이전에 정의되었다고 가정
except ImportError:
    print("FATAL ERROR: ROS libraries must be installed and sourced.")
    sys.exit(1)

# --- 헬퍼 함수: IMU 데이터 추출 로직 ---
# Mono 센서(Color)의 시간과 동기화될 Accel/Gyro 데이터 토픽을 지정합니다.
DEFAULT_IMU_TOPICS = [
    '/device_0/sensor_2/Accel_0/imu/data', # 가속도계
    '/device_0/sensor_2/Gyro_0/imu/data'   # 자이로스코프
]

FALLBACK_CAMERA_SERIAL = "135122070988"
DEFAULT_TEMP_C = 53.400390625 # IMU 메시지에 온도가 없을 경우 사용할 기본값

def bag_get_camera_serial(bag_path):
    # 하드코딩된 시리얼 반환 로직 (이전 논의 반영)
    return FALLBACK_CAMERA_SERIAL

def bag_get_start_datetime(bag_path):
    """
    ROSbag의 인덱스 시간(start_time)이 비정상적일 경우, 
    Image Metadata 토픽에서 'system_time' 키를 추출하여 실제 시작 시간을 반환합니다.
    """
    metadata_topic = '/device_0/sensor_1/Color_0/image/metadata'
    target_key = 'system_time'
    
    try:
        with rosbag.Bag(str(bag_path), 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[metadata_topic], raw=False):
                time_str = None
                if hasattr(msg, 'key') and msg.key == target_key:
                    time_str = msg.value
                elif hasattr(msg, 'values'):
                    for kv in msg.values:
                        if hasattr(kv, 'key') and kv.key == target_key:
                            time_str = kv.value
                            break
                
                if time_str:
                    time_value_ms = float(time_str)
                    start_time_seconds = time_value_ms / 1000.0
                    return datetime.datetime.fromtimestamp(start_time_seconds)
            
            # 메타데이터 검색 실패 시, 파일 수정 시간을 최종 백업으로 사용
            mtime = pathlib.Path(bag_path).stat().st_mtime
            return datetime.datetime.fromtimestamp(mtime)

    except Exception as e:
        # 예외 발생 시, 파일 수정 시간을 최종 백업으로 사용
        mtime = pathlib.Path(bag_path).stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime)


def extract_realsense_imu_to_json(bag_path: pathlib.Path, json_path: pathlib.Path, imu_topics: list = DEFAULT_IMU_TOPICS):
    """
    BAG 파일에서 Accel/Gyro 토픽을 읽어 GoPro 스타일의 JSON 형식으로 통합 저장합니다.
    """
    if json_path.is_file():
        return 0 

    all_samples = []
    
    try:
        # 파일의 절대 시작 시간 기준점 설정 (datetime 객체)
        bag_start_dt = bag_get_start_datetime(bag_path)
        bag_start_time_sec = bag_start_dt.timestamp()
        
        with rosbag.Bag(str(bag_path), 'r') as bag:
            
            for topic, msg, t in bag.read_messages(topics=imu_topics):
                if msg._type != Imu._type:
                    continue
                
                timestamp_sec = msg.header.stamp.to_sec()
                
                # 1. Capture Time Stamp (cts) in milliseconds (상대 시간)
                # 이 시점에서 음수 값이 나올 수 있습니다.
                cts_ms = (timestamp_sec - bag_start_time_sec) * 1000.0
                
                # 2. Date string (ISO 8601 UTC format)
                date_dt = datetime.datetime.fromtimestamp(timestamp_sec, datetime.timezone.utc)
                date_str = date_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
                
                sample = {
                    'value': [],
                    'cts': cts_ms,
                    'date': date_str,
                    'temperature [°C]': DEFAULT_TEMP_C 
                }

                if 'Accel' in topic:
                    sample['value'] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                
                elif 'Gyro' in topic:
                    sample['value'] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

                all_samples.append({'sample': sample, 'topic': topic})


        # --- CTS 보정 로직 (음수 제거 및 기준점 0 설정) ---
        min_cts = min(item['sample']['cts'] for item in all_samples) if all_samples else 0
        cts_offset = 0.0
        if min_cts < 0:
            cts_offset = abs(min_cts)
            print(f"INFO: Adjusting CTS by +{cts_offset:.3f}ms to set reference time to 0.")

        # 4. 최종 JSON 구조 구축 및 CTS 오프셋 적용
        accel_samples = []
        gyro_samples = []
        
        for item in all_samples:
            sample = item['sample']
            topic = item['topic']
            
            # 오프셋 적용
            sample['cts'] += cts_offset

            if 'Accel' in topic:
                accel_samples.append(sample)
            elif 'Gyro' in topic:
                gyro_samples.append(sample)

        # 5. JSON 저장
        total_extracted = len(accel_samples) + len(gyro_samples)
        
        if accel_samples or gyro_samples:
            final_json_data = {
                "1": {
                    "streams": {
                        "ACCL": { 
                            "samples": accel_samples,
                            "name": "Accelerometer",
                            "units": "m/s2"
                        },
                        "GYRO": { 
                            "samples": gyro_samples,
                            "name": "Gyroscope",
                            "units": "rad/s"
                        }
                    },
                    "device name": "Intel RealSense D435I (Mono IMU)", 
                    "frames/second": 400.0 # IMU는 보통 400 FPS이므로 기본값을 400으로 변경
                }
            }
            
            print(f"INFO: Successfully processed {total_extracted} messages into {len(accel_samples)} Accel and {len(gyro_samples)} Gyro entries.")
            
            # ensure_ascii=False 설정으로 유니코드 문자(°C)가 깨지지 않도록 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_json_data, f, indent=4, ensure_ascii=False)
            return total_extracted
        else:
            return 0

    except Exception as e:
        print(f"Error processing {bag_path.name}: {e}")
        return -1 

# --- CLI 명령 정의 (main 함수) ---
@click.command()
@click.option('-t', '--imu_topic', multiple=True, 
              default=DEFAULT_IMU_TOPICS, 
              help="RealSense IMU topics to extract (Accel and Gyro are recommended).")
@click.option('-n', '--num_workers', type=int, default=None, help="Number of concurrent processes.")
@click.argument('session_dir', nargs=-1)
def main(imu_topic, num_workers, session_dir):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    imu_topics_to_check = list(imu_topic) 

    # --- 환경 경로 설정 유지 (보일러플레이트) ---
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    # ---------------------------------------------
    

    for session in session_dir:
        input_dir = pathlib.Path(os.path.expanduser(session)).joinpath('demos')
        input_bag_dirs = [x.parent for x in input_dir.glob('*/raw_bag.bag')]
        
        print(f'Found {len(input_bag_dirs)} BAG directories for IMU extraction.')

        with tqdm(total=len(input_bag_dirs)) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                
                for bag_dir in tqdm(input_bag_dirs, desc="Scheduling IMU extraction"):
                    bag_dir = bag_dir.absolute()
                    
                    bag_path = bag_dir.joinpath('raw_bag.bag')
                    json_path = bag_dir.joinpath('imu_data.json')

                    # 덮어쓰기 방지 로직이 제거됨

                    if len(futures) >= num_workers:
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))

                    futures.add(executor.submit(
                        extract_realsense_imu_to_json, bag_path, json_path, imu_topics_to_check))

                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))

        results = [x.result() for x in completed if x.result() != 0]
        
        print("Done! Summary:")
        print(f"  Total processed files: {len(results)}")
        print(f"  Files with errors: {results.count(-1)}")

# %%
if __name__ == "__main__":
    main()