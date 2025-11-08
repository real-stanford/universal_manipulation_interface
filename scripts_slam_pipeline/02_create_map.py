import os
import pathlib
import subprocess
import sys

# ORB-SLAM3 빌드 후 실행 파일의 상대 경로
EXECUTABLE_PATH = pathlib.Path('../ORB_SLAM3/Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i_v2').resolve()

try:
    # 실행 파일이 존재하는지 최종 확인
    if not EXECUTABLE_PATH.is_file():
        print(f"ERROR: Executable not found at: {EXECUTABLE_PATH}")
        sys.exit(1)
        
    # 실행 (인자 없이 호출)
    subprocess.run([str(EXECUTABLE_PATH)], check=False)
    
except Exception as e:
    print(f"Execution failed: {e}")