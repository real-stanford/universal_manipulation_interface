import sys
import os
import pathlib
import click
import multiprocessing
import concurrent.futures
import json
from tqdm import tqdm
import datetime
import shutil
import numpy as np

# ROS/IMU 의존성 (실제 ROS 환경에 맞게 설치 및 구성 필수)
try:
    import rosbag
    import rospy
    from sensor_msgs.msg import Imu, Image
    from diagnostic_msgs.msg import KeyValue
    from cv_bridge import CvBridge  # 이미지 변환에 필요
    import cv2  # 이미지 저장에 필요
except ImportError:
    print(
        "FATAL ERROR: ROS libraries (rosbag, rospy, cv_bridge, cv2) must be installed and sourced."
    )
    sys.exit(1)

# --- 상수 정의 ---
DEFAULT_IMU_TOPICS = [
    "/device_0/sensor_2/Accel_0/imu/data",
    "/device_0/sensor_2/Gyro_0/imu/data",
]
DEFAULT_COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEFAULT_DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"

DEFAULT_TEMP_C = 53.400390625

# CvBridge 인스턴스 (전역에서 한 번 생성)
try:
    bridge = CvBridge()
except:
    # ROS 환경이 아닐 경우 예외 처리
    bridge = None


# ----------------------------------------------------------------------
## 헬퍼 함수
# ----------------------------------------------------------------------


def bag_get_camera_serial(bag_path):
    return "135122070988"


def bag_get_start_datetime(bag_path):
    """
    Extracts system_time from Image Metadata topic for accurate start time.
    Falls back to file modification time if metadata is not found.
    """
    metadata_topic = "/device_0/sensor_1/Color_0/image/metadata"
    target_key = "system_time"

    # ----------------------------------------
    # [수정] try/except 블록 구조 명확화
    # ----------------------------------------
    try:
        with rosbag.Bag(str(bag_path), "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[metadata_topic], raw=False):
                time_str = None
                if hasattr(msg, "key") and msg.key == target_key:
                    time_str = msg.value
                elif hasattr(msg, "values"):
                    for kv in msg.values:
                        if hasattr(kv, "key") and kv.key == target_key:
                            time_str = kv.value
                            break

                if time_str:
                    time_value_ms = float(time_str)
                    start_time_seconds = time_value_ms / 1000.0
                    return datetime.datetime.fromtimestamp(start_time_seconds)

            # 루프가 끝나도 시간을 찾지 못한 경우
            mtime = pathlib.Path(bag_path).stat().st_mtime
            return datetime.datetime.fromtimestamp(mtime)

    except Exception as e:
        # print(f"Error reading metadata time: {e}")
        pass  # 오류 발생 시 아래의 최종 fallback 로직으로 이동

    # 최종 fallback: 파일 수정 시간을 사용
    mtime = pathlib.Path(bag_path).stat().st_mtime
    return datetime.datetime.fromtimestamp(mtime)


def extract_realsense_imu_to_json(
    bag_path: pathlib.Path,
    json_path: pathlib.Path,
    imu_topics: list = DEFAULT_IMU_TOPICS,
):
    all_samples = []

    # ----------------------------------------
    # [수정] try/except 블록 구조 명확화
    # ----------------------------------------
    try:
        bag_start_dt = bag_get_start_datetime(bag_path)
        bag_start_time_sec = bag_start_dt.timestamp()

        with rosbag.Bag(str(bag_path), "r") as bag:
            for topic, msg, t in bag.read_messages(topics=imu_topics):
                if msg._type != "sensor_msgs/Imu":
                    continue

                timestamp_sec = msg.header.stamp.to_sec()
                cts_ms = (timestamp_sec - bag_start_time_sec) * 1000.0
                date_dt = datetime.datetime.fromtimestamp(
                    timestamp_sec, datetime.timezone.utc
                )
                date_str = date_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

                sample = {
                    "value": [],
                    "cts": cts_ms,
                    "date": date_str,
                    "temperature [°C]": DEFAULT_TEMP_C,
                }

                if "Accel" in topic:
                    sample["value"] = [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                    ]
                elif "Gyro" in topic:
                    sample["value"] = [
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ]

                all_samples.append({"sample": sample, "topic": topic})

        # CTS 보정 로직 (상대 시간 기준점을 0으로 설정)
        min_cts = (
            min(item["sample"]["cts"] for item in all_samples) if all_samples else 0
        )
        cts_offset = 0.0
        if min_cts < 0:
            cts_offset = abs(min_cts)

        accel_samples = []
        gyro_samples = []

        for item in all_samples:
            sample = item["sample"]
            topic = item["topic"]
            sample["cts"] += cts_offset

            if "Accel" in topic:
                accel_samples.append(sample)
            elif "Gyro" in topic:
                gyro_samples.append(sample)

        total_extracted = len(accel_samples) + len(gyro_samples)

        if accel_samples or gyro_samples:
            # 사용자 요청에 따라 평면 구조로 시리얼라이즈
            final_json_data = {"ACCL": accel_samples, "GYRO": gyro_samples}

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(final_json_data, f, indent=4, ensure_ascii=False)
            return total_extracted
        else:
            return 0

    except Exception as e:
        print(f"Error processing {bag_path.name} (IMU): {e}")
        return -1


@click.command()
@click.option(
    "-t",
    "--imu_topic",
    multiple=True,
    default=DEFAULT_IMU_TOPICS,
    help="RealSense IMU topics to extract (Accel and Gyro are recommended).",
)
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
def main(imu_topic, color_topic, depth_topic, num_workers, session_dir):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    imu_topics_to_check = list(imu_topic)

    # --- 환경 경로 설정 유지 (보일러플레이트) ---
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    # ------------------------------------------

    # 메시지 업데이트: IMU 데이터 추출만 수행함을 명시
    print("--- Starting RealSense BAG IMU Data Extraction ---")

    for session in session_dir:
        session_path = pathlib.Path(os.path.expanduser(session)).absolute()
        input_bag_paths = [x for x in session_path.glob("**/raw_bag.bag")]

        if not input_bag_paths:
            print(
                f"Warning: No 'raw_bag.bag' found in directories under {session_path}. Skipping."
            )
            continue

        # 메시지 업데이트: IMU 추출만 수행함을 명시
        print(f"Found {len(input_bag_paths)} BAG files for IMU extraction.")

        # IMU 추출 (1) = 총 1 작업으로 변경
        total_tasks = len(input_bag_paths) * 1

        # ProcessPoolExecutor를 사용하여 병렬 처리 (CPU 바인딩 작업이므로 프로세스 풀 사용)
        with tqdm(total=total_tasks) as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = set()

                # 설명 업데이트: IMU 추출만 예약
                for bag_path in tqdm(input_bag_paths, desc="Scheduling IMU extraction"):
                    bag_dir = bag_path.parent
                    imu_json_path = bag_dir.joinpath("imu_data.json")

                    # 1. IMU 추출 작업 예약 (JSON)
                    imu_future = executor.submit(
                        extract_realsense_imu_to_json,
                        bag_path,
                        imu_json_path,
                        imu_topics_to_check,
                    )
                    futures.add(imu_future)

                    # 2. RGB/Depth 프레임 및 타임스탬프 추출 작업 예약 (제거됨)

                    # 완료된 작업 처리 및 tqdm 업데이트
                    if len(futures) >= num_workers:
                        completed, futures = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        pbar.update(len(completed))

                # 남아있는 모든 작업 완료 대기
                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))

        # 결과 요약
        results = [
            x.result() for x in completed if x.result() != 0 and x.result() != -1
        ]
        errors = [x.result() for x in completed if x.result() == -1]

        print("\nDone! Summary:")
        # 결과 메시지 업데이트
        print(f"  Total successful IMU extractions: {len(results)}")
        print(f"  Files with errors: {len(errors)}")


# %%
if __name__ == "__main__":
    main()
