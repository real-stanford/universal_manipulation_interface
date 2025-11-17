# python3 ~/Desktop/study/universal_manipulation_interface/scripts/detect_aruco.py\
#  --input ~/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/raw_video.mp4\
#  --output ~/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/tag_detection.pkl\
#  --intrinsics_json ~/Desktop/study/universal_manipulation_interface/example/calibration/realsense_intrinsics.json\
#  --aruco_yaml ~/Desktop/study/universal_manipulation_interface/example/calibration/aruco_config.yaml



python3 ~/Desktop/study/universal_manipulation_interface/scripts/detect_aruco.py\
 --input ~/Documents/1920_gripper_30hz.mp4\
 --output ~/Documents/tag_detection.pkl\
 --intrinsics_json ~/Desktop/study/universal_manipulation_interface/example/calibration/realsense_intrinsics.json\
 --aruco_yaml ~/Desktop/study/universal_manipulation_interface/example/calibration/aruco_config.yaml


python3 scripts/calibrate_gripper_range.py --input ./my_example_demo_session/demos/mapping/tag_detection.pkl --output ./test.json