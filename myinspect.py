import pickle
import numpy as np
import pandas as pd
import sys
import os

# 경로를 본인 환경에 맞게 수정하세요
pkl_path = "/home/sungjoon/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/tag_detection.pkl"
csv_path = "/home/sungjoon/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/camera_trajectory.csv"

def inspect_data(pkl_path, csv_path):
    # 1. ArUco 데이터 확인 (PKL)
    if not os.path.exists(pkl_path):
        print("Error: PKL file not found.")
        return

    with open(pkl_path, 'rb') as f:
        tag_data = pickle.load(f)
    
    print(f"--- [ArUco Data Analysis] ---")
    print(f"Total Frames: {len(tag_data)}")
    
    # 앞부분 5개 데이터 확인
    print("\nFirst 5 Frames of ArUco Data:")
    for i in range(min(5, len(tag_data))):
        frame = tag_data[i]
        t = frame.get('time', 0)
        tags = frame.get('tag_dict', {})
        print(f"Frame {i}: Time={t:.6f}, Num Tags={len(tags)}")
        
        # 태그 위치값(tvec)이 변하는지 확인
        for tag_id, tag_info in tags.items():
            tvec = tag_info['tvec']
            print(f"  - Tag {tag_id} tvec: {tvec.flatten()}")

    # 2. SLAM 데이터 확인 (CSV)
    if os.path.exists(csv_path):
        print(f"\n--- [SLAM Trajectory Analysis] ---")
        df = pd.read_csv(csv_path)
        slam_times = df.iloc[:, 0].values # 보통 첫 번째 컬럼이 시간
        print(f"SLAM Start Time: {slam_times[0]:.6f}")
        print(f"SLAM End Time  : {slam_times[-1]:.6f}")
        
        # 3. 시간 비교
        aruco_start = tag_data[0]['time']
        print(f"\n--- [Comparison] ---")
        print(f"ArUco Start: {aruco_start:.6f}")
        print(f"SLAM Start : {slam_times[0]:.6f}")
        print(f"Diff       : {abs(aruco_start - slam_times[0]):.6f} sec")
        
        if abs(aruco_start - slam_times[0]) > 10.0:
            print("\n🚨🚨 FATAL ERROR: Timestamps are NOT synced! 🚨🚨")
            print("The gap is huge. Interpolation will fail and return constant values.")
    else:
        print("CSV path not found, skipping comparison.")

if __name__ == "__main__":
    inspect_data(pkl_path, csv_path)