# python3 ~/Desktop/study/universal_manipulation_interface/scripts/detect_aruco.py\
#  --input ~/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/raw_video.mp4\
#  --output ~/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/tag_detection.pkl\
#  --intrinsics_json ~/Desktop/study/universal_manipulation_interface/example/calibration/realsense_intrinsics.json\
#  --aruco_yaml ~/Desktop/study/universal_manipulation_interface/example/calibration/aruco_config.yaml



python3 ~/Desktop/study/universal_manipulation_interface/scripts/detect_aruco.py\
 --input /home/sungjoon/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/gripper_calibration_135122070988_2025.11.24_19.00.18.856155/raw_video.mp4\
 --output ~/Documents/tag_detection.pkl\
 --intrinsics_json ~/Desktop/study/universal_manipulation_interface/example/calibration/realsense_intrinsics.json\
 --aruco_yaml ~/Desktop/study/universal_manipulation_interface/example/calibration/aruco_config.yaml


python3 scripts/calibrate_gripper_range.py\
 --input /home/sungjoon/Documents/tag_detection.pkl\
 --output /home/sungjoon/Documents/test.json

