import pyrealsense2 as rs

bag_path = "/home/sungjoon/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/raw_bag.bag"  # Bag 파일 경로

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_path, repeat_playback=False)

# 파이프라인 시작 (이때 내부적으로 Bag의 토픽 정보를 읽어옴)
profile = pipe.start(cfg)

# Color 스트림 찾기 (이게 '/device_0/sensor_1/Color_0'에 해당)
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()

print(f"\n=== Camera Info (Color Raw) ===")
print(intr)

print(f"\n=== Camera Info (Color) ===")
print(f"Width: {intr.width}, Height: {intr.height}")
print(f"Distortion Model: {intr.model}")
print(f"D (Distortion Coeffs): {intr.coeffs}")
# K Matrix (3x3) 순서: [fx, 0, ppx, 0, fy, ppy, 0, 0, 1]
print(f"K (Intrinsics Matrix):")
print(f"[{intr.fx}, 0.0, {intr.ppx}]")
print(f"[0.0, {intr.fy}, {intr.ppy}]")
print(f"[0.0, 0.0, 1.0]")

pipe.stop()