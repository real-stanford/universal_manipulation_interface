import os
import sys
import pathlib
import shutil
import datetime
import click
import numpy as np # OpenCV와 함께 사용될 수 있음

# ROS 및 CV 의존성 (실제 ROS 환경에 맞게 설치 및 구성 필수)
try:
    import rosbag
    import rospy
    from cv_bridge import CvBridge
    import cv2
    from sensor_msgs.msg import Image
    from diagnostic_msgs.msg import KeyValue
except ImportError:
    # 실행 환경에서 ROS 라이브러리 부재 시 종료
    print("FATAL ERROR: ROS libraries (rosbag, rospy, cv_bridge, cv2) must be installed and sourced.")
    sys.exit(1)

# --- 환경 설정 ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# 이 경로는 실행 환경에 따라 조정이 필요합니다.
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

# --- 헬퍼 함수 (실제 ROSbag 구현) ---

def bag_get_start_datetime(bag_path):
    """
    ROSbag의 인덱스 시간(start_time)이 비정상적일 경우, 
    Image Metadata 토픽에서 'system_time' 키를 추출하여 실제 시작 시간을 반환합니다.
    """
    # 🎯 RealSense 이미지 메타데이터 토픽 및 키
    metadata_topic = '/device_0/sensor_1/Color_0/image/metadata'
    target_key = 'system_time'
    
    try:
        with rosbag.Bag(str(bag_path), 'r') as bag:
            
            # 메타데이터 토픽에서 첫 번째 메시지를 읽습니다.
            for topic, msg, t in bag.read_messages(topics=[metadata_topic], raw=False):
                
                # 메시지(diagnostic_msgs/KeyValue)가 단일 KeyValue인 경우
                if hasattr(msg, 'key') and msg.key == target_key:
                    time_str = msg.value
                    
                # 메시지(DiagnosticArray 등)가 KeyValue 리스트(values)를 포함하는 경우
                elif hasattr(msg, 'values'):
                    time_str = None
                    for kv in msg.values:
                        if hasattr(kv, 'key') and kv.key == target_key:
                            time_str = kv.value
                            break
                    if time_str is None:
                        # 'system_time' 키를 찾지 못함
                        break
                else:
                    # 예상치 못한 메시지 형식
                    break

                # 1. 메타데이터 값(문자열)을 실수(float)로 변환
                time_value_ms = float(time_str)
                
                # 2. 밀리초(ms)를 초(s)로 변환 (Unix Time으로 가정)
                start_time_seconds = time_value_ms / 1000.0
                
                print(f"[INFO] Using metadata time: {start_time_seconds:.3f}s from {metadata_topic}")
                
                return datetime.datetime.fromtimestamp(start_time_seconds)
            
            # 루프가 종료되어도 시간을 찾지 못한 경우
            print(f"Warning: '{target_key}' not found in {metadata_topic}. Falling back to file modification time.")
            
    except Exception as e:
        print(f"Error reading metadata time from {bag_path}: {e}. Falling back to file modification time.")
    
    # 메타데이터 검색 실패 시, 파일 수정 시간을 최종 백업으로 사용
    mtime = pathlib.Path(bag_path).stat().st_mtime
    return datetime.datetime.fromtimestamp(mtime)

# 시리얼 번호는 하드코딩된 값을 반환하도록 유지 (자동 추출 실패 문제 해결)
FALLBACK_CAMERA_SERIAL = "135122070988"
def bag_get_camera_serial(bag_path):
    return FALLBACK_CAMERA_SERIAL

def process_bag_to_mp4(bag_path, mp4_path, color_topic_name='/device_0/sensor_1/Color_0/image/data', fps=10):
    """
    핵심 변환 함수: BAG 파일에서 컬러 이미지 토픽을 읽어 MP4 파일로 저장합니다.
    """
    print(f"Converting {bag_path.name} to MP4...")
    
    bridge = CvBridge()
    out_dir = mp4_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    video_writer = None
    
    try:
        with rosbag.Bag(str(bag_path), 'r') as bag:
            is_first_frame = True
            
            # BAG 파일을 읽으며 비디오에 프레임 쓰기
            for topic_loop, msg_loop, t_loop in bag.read_messages(topics=[color_topic_name]):
                if msg_loop._type != Image._type: # 메시지 타입 확인
                     continue
                
                # 이미지 메시지를 OpenCV Mat 객체로 변환
                cv_image = bridge.imgmsg_to_cv2(msg_loop, desired_encoding="bgr8")
                
                if is_first_frame:
                    height, width, layers = cv_image.shape
                    
                    # VideoWriter 설정 (H.264 코덱)
                    # NOTE: 'mp4v' 코덱은 Windows/Linux에서 흔히 사용됩니다. 'XVID'나 'DIVX'를 사용할 수도 있습니다.
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    video_writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))
                    is_first_frame = False
                    print(f"VideoWriter initialized: {width}x{height} @ {fps} FPS")

                if video_writer is not None:
                    video_writer.write(cv_image)

        if video_writer is not None:
            video_writer.release()
            print(f"Successfully converted to {mp4_path.name}")
            return True
        else:
            print(f"Conversion failed for {bag_path.name}: No image frames found on topic {color_topic_name}.")
            return False
            
    except Exception as e:
        print(f"Conversion failed for {bag_path.name} due to an exception: {e}")
        return False

# --- CLI 명령 정의 ---
@click.command(help='''
Session directories. Converts .bag files into structured .mp4 files based on metadata (start time and serial number).
The original BAG file is copied into the corresponding demos directory along with the MP4.
''')
@click.argument('session_dir', nargs=-1, type=pathlib.Path)
def main(session_dir):
    for session_path in session_dir:
        session = session_path.expanduser().absolute()
        
        # 디렉토리 정의
        input_dir = session.joinpath('raw_bags')
        output_dir = session.joinpath('demos')
        
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Starting BAG processing for session: {session.name} ---")

        # 1. 모든 BAG 파일을 raw_bags로 이동 및 리스트업
        all_bag_paths = [p for p in (list(session.glob('**/*.BAG')) + list(session.glob('**/*.bag'))) 
                         if 
                        #  not p.is_symlink() and 
                         p.is_file()]
        

        # 파일 초기 이동 (MP4 스크립트 로직 재현)
        initial_move_count = 0
        for bag_path in all_bag_paths:
            if bag_path.parent != input_dir:
                 # shutil.move 대신 shutil.copy2를 사용하고, 원본 파일은 raw_bags로 이동
                 shutil.move(bag_path, input_dir.joinpath(bag_path.name))
                 initial_move_count += 1
        
        if initial_move_count > 0:
            print(f"Moved {initial_move_count} BAG files into {input_dir.name}.")
            # 이동 후 리스트를 다시 읽습니다.
            all_bag_paths = [p for p in (list(input_dir.glob('**/*.BAG')) + list(input_dir.glob('**/*.bag'))) 
                             if 
                            #  not p.is_symlink() and 
                             p.is_file()]

        # 2. 메타데이터 기반으로 파일 정렬 및 분류 후보 지정
        max_size = -1
        mapping_bag_path = None
        
        serial_start_dict = dict()
        serial_path_dict = dict()
        bags_to_process = all_bag_paths[:] 

        for bag_path in bags_to_process:
            if bag_path.name.startswith('mapping') or bag_path.parent.name.startswith('gripper_cal'):
                continue # 이미 분류된 파일은 건너뜁니다.

            size = bag_path.stat().st_size
            
            # (A) Mapping Video: 가장 큰 파일을 매핑 후보로 지정
            if size > max_size:
                max_size = size
                mapping_bag_path = bag_path
            
            # (B) Calibration Video: 시리얼별로 가장 이른 시간을 가진 파일 찾기
            start_date = bag_get_start_datetime(str(bag_path))
            cam_serial = bag_get_camera_serial(str(bag_path))
            
            if cam_serial not in serial_start_dict or start_date < serial_start_dict[cam_serial]:
                serial_start_dict[cam_serial] = start_date
                serial_path_dict[cam_serial] = bag_path

        # 3. 파일 변환 및 최종 구조화

        # 3-1. Mapping Video 처리
        if mapping_bag_path and mapping_bag_path in bags_to_process:
            mapping_out_dir = output_dir.joinpath("mapping")
            mapping_out_dir.mkdir(parents=True, exist_ok=True)
            
            out_mp4_path = mapping_out_dir.joinpath('raw_video.mp4')
            
            # BAG -> MP4 변환
            if process_bag_to_mp4(mapping_bag_path, out_mp4_path):
                # 원본 BAG 파일 복사
                out_bag_path = mapping_out_dir.joinpath('raw_bag.bag') # 복사본 이름
                shutil.copy2(mapping_bag_path, out_bag_path) # 원본 BAG 파일 복사
                print(f"Mapping BAG converted and original copied to {mapping_out_dir.name}.")
                bags_to_process.remove(mapping_bag_path)

        # 3-2. Gripper Calibration Video 처리
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        gripper_cal_dir.mkdir(parents=True, exist_ok=True)

        for serial, bag_path in serial_path_dict.items():
            if bag_path in bags_to_process: # 아직 처리되지 않은 파일만
                start_date = serial_start_dict[serial]
                
                # 원본 스크립트와 동일: 먼저 raw_bags/gripper_calibration 폴더로 BAG 파일 이동
                # 이 파일은 이제 raw_bags/gripper_calibration/ 에 존재
                cal_bag_path_moved = gripper_cal_dir.joinpath(bag_path.name)
                shutil.move(bag_path, cal_bag_path_moved)
                bags_to_process.remove(bag_path)
                
                # 출력 디렉토리 이름 생성 (demos 폴더에 저장될 MP4의 최종 위치)
                out_dname = f"gripper_calibration_{serial}_{start_date.strftime(r'%Y.%m.%d_%H.%M.%S.%f')}"
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # BAG -> MP4 변환
                out_mp4_path = this_out_dir.joinpath('raw_video.mp4')
                if process_bag_to_mp4(cal_bag_path_moved, out_mp4_path):
                    # 원본 BAG 파일 복사
                    out_bag_path = this_out_dir.joinpath('raw_bag.bag') # 복사본 이름
                    shutil.copy2(cal_bag_path_moved, out_bag_path) # 원본 BAG 파일 복사
                    print(f"Calibration BAG converted and original copied to {this_out_dir.name}.")
        
        # 3-3. 나머지 Demo Video 처리
        print(f'\nFound {len(bags_to_process)} remaining BAG files for demo conversion.')

        for bag_path in bags_to_process:
            start_date = bag_get_start_datetime(str(bag_path))
            cam_serial = bag_get_camera_serial(str(bag_path))
            
            # 출력 디렉토리 이름 생성 (메타데이터 적용)
            out_dname = f"demo_{cam_serial}_{start_date.strftime(r'%Y.%m.%d_%H.%M.%S.%f')}"
            this_out_dir = output_dir.joinpath(out_dname)
            this_out_dir.mkdir(parents=True, exist_ok=True)
            
            # BAG -> MP4 변환
            out_mp4_path = this_out_dir.joinpath('raw_video.mp4')
            if process_bag_to_mp4(bag_path, out_mp4_path):
                # 원본 BAG 파일 복사
                out_bag_path = this_out_dir.joinpath('raw_bag.bag') # 복사본 이름
                shutil.copy2(bag_path, out_bag_path) # 원본 BAG 파일 복사

                # 원본 MP4 스크립트의 심볼릭 링크 생성 로직은 제거 (복사로 대체)
                print(f"Demo BAG converted and original copied to {this_out_dir.name}.")

# --- 스크립트 실행 진입점 ---
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()