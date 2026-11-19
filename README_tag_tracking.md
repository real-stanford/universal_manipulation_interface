# Tag tracking pipeline (GTSAM)

Gripper poses from a head-mounted GoPro tracking ArUco cubes on the grippers,
instead of wrist camera SLAM. Continuous takes are split into episodes between
dock periods; the docking is removed and the wrist camera clocks are
synchronized to the head camera from motion. Output: `dataset_plan.pkl` (same format as
the SLAM pipeline) and one MCAP file per episode.

## Hardware

- One head GoPro and one GoPro per gripper (wrist), same resolution and frame rate.
- One 70 mm cube per gripper with four 60 mm tags, rigidly mounted to the gripper.
- Environment tags on the table and walls (static); one of them is the world origin (`anchor_tag`).
- Two environment tags next to where the grippers rest between episodes (`dock_tags`).

## Install

```
mamba env create -f conda_environment.yaml
conda activate umi
pip install gtsam==4.2 mcap==1.6.0
docker pull chicheng/openicc:latest
```

## Calibration files

`example/tag_tracking/` holds the calibration of our setup; all paths in the configs are relative to it.

| File | Content |
|---|---|
| `tag_tracking.yaml` | cameras by serial, bodies, episodes; format in `umi/tag_tracking/config.py` |
| `aruco_config.yaml` | dictionary and tag sizes (black square edge, m) |
| `gripper_config.yaml` | finger tags and TCP geometry per wrist camera; format in `umi/common/gripper_config_util.py` |
| `corner_noise.json` | tag corner noise (step 1) |
| `setup_20261116/` | setup calibration (step 2): tag map, cube shapes, cube -> wrist -> TCP |
| `openicc/<camera>/cam/` | OpenICC intrinsics |
| `openicc/<camera>/cam_imu/` | OpenICC camera-IMU calibration (fitted trajectory removed) |
| `*_setting_720*.yaml` | IMU noise per camera |
| `gopro_intrinsics_2_7k.json` | fallback intrinsics |

Cameras (GoPro HERO12, Wide, 2.7K 4:3, 59.94 fps, stabilization off):

| Serial | Camera | OpenICC |
|---|---|---|
| C3501326502664 | head | `openicc/head` |
| C3501326029433 | right gripper (robot0) | `openicc/gripper2` |
| C3501326502891 | left gripper (robot1) | `openicc/gripper1` |

| C3501326500348 | right gripper at station1 | `openicc/cam376` |

With other cameras, calibrate them (Camera calibration below) and replace the serials,
intrinsics and camera-IMU files. Steps 1 and 2 below are done for this setup; redo step 2
after moving environment tags or remounting a cube or camera.

## Camera calibration

Once per camera (and after changing its lens or video settings): intrinsics (Kannala-Brandt
fisheye) and camera-IMU rotation / translation with OpenImuCameraCalibrator (OpenICC).

Board: ChArUco 12 x 9 squares, 30 mm squares, 22 mm markers, `DICT_5X5_100`, flat and rigid.

Record with the camera's take settings (2.7K 4:3, 59.94 fps, Wide, stabilization off), three videos:

| Video | How |
|---|---|
| `cam` | 30-60 s, move slowly around the board: all image regions, near and far, tilted views |
| `cam_imu` | 40-60 s, board in view, fast rotations and translations on all three axes, good light |
| `imu_bias` | 60 s, camera lying still |

OpenICC image with the configurable board (once): OpenICC with the
`aruco_marker_length_m` / `aruco_dict` options, built on top of `chicheng/openicc:latest`:

```
git clone -b configurable-charuco-board https://github.com/PushpakAg/OpenImuCameraCalibrator.git
cd OpenImuCameraCalibrator
docker build -f Dockerfile.overlay -t openicc:configurable-board .
```

Sort a camera's three videos into the OpenICC layout (roles from the gyro: the still video, then
the board video with less / more rotation); check the table with `--dry_run` first:

```
python scripts/prepare_openicc_dataset.py -o ~/openicc_calib -n set2_head --dry_run \
    ~/Videos/set2_calib/head/GX010157.MP4 ~/Videos/set2_calib/head/GX010158.MP4 ~/Videos/set2_calib/head/GX010159.MP4
python scripts/prepare_openicc_dataset.py -o ~/openicc_calib -n set2_head \
    ~/Videos/set2_calib/head/GX010157.MP4 ~/Videos/set2_calib/head/GX010158.MP4 ~/Videos/set2_calib/head/GX010159.MP4
```

Calibrate (15-25 min per camera; one terminal per camera to run several at once):

```
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v ~/openicc_calib/set2_head:/dataset \
    openicc:configurable-board python3 /OpenImuCameraCalibrator/python/run_gopro_calibration.py \
    --path_calib_dataset=/dataset --path_to_build=/OpenImuCameraCalibrator/build/applications \
    --num_squares_x=12 --num_squares_y=9 --checker_size_m=0.030 --marker_size_m=0.022 --aruco_dict=DICT_5X5_100 \
    --camera_model=FISHEYE --image_downsample_factor=1 --bias_calib_remove_s=2 \
    2>&1 | tee ~/openicc_calib/set2_head/calibration.log
```

Import into `example/tag_tracking/` (intrinsics, camera-IMU without the fitted trajectory, IMU
noise setting); prints the numbers to check and the config lines:

```
python scripts/import_openicc_calibration.py -i ~/openicc_calib/set2_head -n set2_head -s C3501326495617
```

Check: intrinsics reprojection error about 0.3-1 px; camera-IMU translation about 20-25 mm
from zero (the IMU sits at the same place in every HERO12). A translation near 0 means the
`cam_imu` video moved too little: record it again with faster translations. Then add the
camera to the tag tracking config (`cameras`, `gripper_imu` or `head_imu`).

Print tags (black square `-t` mm, as in `aruco_config.yaml`):

```
python scripts/gen_tag_sticker_pdfs.py -o print/env_tags --ids 14-17,20-23,26-29,32-35,38-41,44-47 -t 80
```

## Session layout

```
<session>/raw_videos/*.MP4                      head and wrist videos of the take(s)
<session>/raw_videos/gripper_calibration/*.MP4  one open/close video per wrist camera
<session>/raw_videos/head_mapping/*.MP4         head camera walk past all environment tags
```

## Recording

- Gripper calibration: per gripper, open and close the fingers fully a few times.
- Head mapping: walk the head camera slowly past every environment tag.
- Take: start all cameras, then repeat per episode: grippers on the dock, still for 1 s
  with the head camera looking at the dock tags; do the task; return to the dock.

## Run

### 1. Corner noise (once per camera model)

From a video of a static camera looking at static tags:

```
python scripts_tag_tracking/01_measure_corner_noise.py -i <static_video.MP4> -c example/tag_tracking/tag_tracking.yaml
```

### 2. Setup calibration (once per setup)

On one session, with `setup_calibration` removed from `tag_tracking.yaml`:

```
python run_tag_tracking_pipeline.py -c example/tag_tracking/tag_tracking.yaml --stop_after segment_episodes <session>
python scripts_tag_tracking/check_wrist_events.py -i <session> -c example/tag_tracking/tag_tracking.yaml
python scripts_tag_tracking/calibrate_setup.py -i <session> -c example/tag_tracking/tag_tracking.yaml -o example/tag_tracking/setup_01 -j 24
```

`check_wrist_events.py` plots the episode detection to `<session>/tag_tracking/wrist_events/`;
tune the `episodes` section if episodes are missed.

Then set in `tag_tracking.yaml`:

```yaml
setup_calibration: setup_01
```

After changing only the TCP geometry in `gripper_config.yaml`:

```
python scripts_tag_tracking/calibrate_setup.py -c example/tag_tracking/tag_tracking.yaml -o example/tag_tracking/setup_20261116 --tcp_only
```

### 3. Sessions

```
python run_tag_tracking_pipeline.py -c example/tag_tracking/tag_tracking.yaml -j 16 -nr <session>
python scripts_tag_tracking/07_export_mcap.py -i <session> -j 4
```

| Option | |
|---|---|
| `-j` | parallel processes, default all cores |
| `-nr` | skip the render step |
| `--start_from <step>` / `--stop_after <step>` | run a range of steps |

Steps: `sort_videos imu wrist_tags gripper_range audio_sync head_tags segment_episodes use_setup
calibrate_bodies track_bodies fuse_head_imu trim_docking gripper_tcp sync_wrists fuse_grippers render export_plan`.
Logs: `<session>/logs/`.

After a TCP change, without re-tracking:

```
python scripts_tag_tracking/00c_use_setup.py -i <session> -c example/tag_tracking/tag_tracking.yaml
python scripts_tag_tracking/06_export_dataset_plan.py -i <session> -c example/tag_tracking/tag_tracking.yaml
python scripts_tag_tracking/07_export_mcap.py -i <session> -j 4
```

### 4. Checks

```
python scripts_tag_tracking/check_tcp_frame.py -i <session> -c example/tag_tracking/tag_tracking.yaml
python scripts_tag_tracking/04_render_tracking.py -i <session> -c example/tag_tracking/tag_tracking.yaml
```

### 5. Training dataset

```
python scripts_slam_pipeline/07_generate_replay_buffer.py --no_mirror -o <dataset.zarr.zip> <session>
```

## Outputs

| Path | |
|---|---|
| `<session>/tag_tracking/episodes.json` | episodes per head video (head video seconds), task interval without docking |
| `<session>/tag_tracking/wrist_sync.json` | wrist clock corrections and sync accuracy |
| `<session>/demos/demo_<head>/tag_tracking.pkl` | head, cube and fused gripper poses |
| `<session>/dataset_plan.pkl` | episodes in the UMI dataset plan format |
| `<session>/dataset_mcap/episode_*.mcap` | one file per episode |
| `<session>/tag_tracking/tcp_check_*.png` | TCP check renders |

## MCAP

| Topic | Schema |
|---|---|
| `robot{i}/eef/state` | `umi.msg.EefState`: TCP `pos` (m), `rot_axis_angle` (rad), world frame |
| `robot{i}/gripper/state` | `umi.msg.GripperState`: `width` (m) |
| `robot{i}/eef_demo_start/state`, `robot{i}/eef_demo_end/state` | first / last TCP pose of the episode |
| `robot{i}/imu` | `umi.msg.ImuSample`: wrist camera IMU |
| `/cameras/camera{j}/video` | `foxglove.CompressedVideo`, H.265 from the GoPro files, not re-encoded |

`robot0` right, `robot1` left; `camera0` robot0 wrist, `camera1` robot1 wrist, `camera2` head.
Metadata `recording`: episode length, image size, fps, world frame tag, TCP in the wrist camera frame.

## Limitations

- Episodes with a head tracking gap longer than `max_gap_sec` (0.25 s) inside are not exported to `dataset_plan.pkl`.
- A task ends when the first gripper returns to the dock.
