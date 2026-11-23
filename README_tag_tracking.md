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

## Workflow

For a recording day, in order (details in the sections below):

| Step | What | Once per |
|---|---|---|
| Camera calibration | intrinsics + camera-IMU per GoPro, imported into `example/tag_tracking/` | camera |
| 0. Sort | day folder (local or S3) -> one session per task | recording day |
| 1. Corner noise | tag corner noise model | camera model |
| 2. Station setup | draft configs from a session, setup calibration (tag map, cubes, TCP) | station, or after moving tags / cubes / cameras |
| 3. Sessions | tracking pipeline -> `dataset_plan.pkl` | session |
| 4-6. Checks, dataset, delivery | renders, replay buffer, Odyssey delivery | session |

## End to end example

One recording day of station3 (2026-11-20), from the uploaded GoPro files to exported episodes.
Two locations to replace with your own:
- `s3://my-bucket/UMI_Collection`: where the cameras' files are uploaded, one folder per
  station and day with `head/`, `left/`, `right/` inside (as in Recording below);
- `~/umi_data`: a local folder for the sessions being processed (about 25 GB per task).

All commands from the repository root, in the `umi` environment. The aws CLI must be logged in
(`aws sts get-caller-identity` prints your account; if the token expired, log in again).

**1. Propose the tasks of the day** (reads the video headers and the short clips on S3,
nothing is downloaded; 10-30 min):

```
python scripts_tag_tracking/sort_recordings.py s3://my-bucket/UMI_Collection/station3/2026-11-20 --save ~/umi_data/station3/inventory_2026-11-20.json --propose ~/umi_data/station3/tasks_2026-11-20.yaml
```

Open `~/umi_data/station3/tasks_2026-11-20.yaml`: one entry per task (`task01`, `task02`, ...)
with its takes, gripper calibration and head mapping; the comments say why each short clip was
taken as what, and what was left out. Edit it if something is wrong (recording numbers as
printed by the command).

**2. Create the sessions**: copies each task's videos on S3 to
`industrial_data/<task>/raw_videos/` (with `pairs.csv`); videos already there are skipped, so
it can be rerun:

```
python scripts_tag_tracking/sort_recordings.py s3://my-bucket/UMI_Collection/station3/2026-11-20 --tasks ~/umi_data/station3/tasks_2026-11-20.yaml -o s3://my-bucket/UMI_Collection/station3/industrial_data
```

**3. Download a session:**

```
aws s3 sync s3://my-bucket/UMI_Collection/station3/industrial_data/task02 ~/umi_data/station3/industrial_data/task02
```

**4. Process it** (all steps; the station's config and setup calibration are in the repo;
1-3 h depending on the length and the cores):

```
python run_tag_tracking_pipeline.py -c example/tag_tracking/tag_tracking_station3.yaml -nr ~/umi_data/station3/industrial_data/task02
```

The end of `~/umi_data/station3/industrial_data/task02/logs/16_export_plan.log` gives the yield:
episodes, seconds used, and why episodes were dropped. Episodes with a short tracking gap are
kept (pose marked not available in the gap) by exporting the plan again with `--keep_gaps`:

```
python scripts_tag_tracking/06_export_dataset_plan.py -i ~/umi_data/station3/industrial_data/task02 -c example/tag_tracking/tag_tracking_station3.yaml --keep_gaps
```

**5. Export the episodes.** MCAP files (`<session>/dataset_mcap/`):

```
python scripts_tag_tracking/07_export_mcap.py -i ~/umi_data/station3/industrial_data/task02 -c example/tag_tracking/tag_tracking_station3.yaml -j 4
```

Odyssey delivery: copy `example/tag_tracking/delivery.yaml` to `~/umi_data/station3/delivery.yaml`,
fill in `delivery:` and one entry per session name (`task02:` with `task_text`, `task_family`,
`operator_id`, `station_id`, `outcome`), then:

```
python scripts_tag_tracking/13_export_delivery.py -i ~/umi_data/station3/industrial_data/task02 -c example/tag_tracking/tag_tracking_station3.yaml -d ~/umi_data/station3/delivery.yaml -o ~/umi_data/station3/delivery --no_annotations -j 16
```

Repeat 3-5 for each task, or all tasks of the station in one run (download, pipeline,
`--keep_gaps` plan, MCAP export, upload per task; a failing task does not stop the others; a
summary with the yield per task at the end):

```
python scripts_tag_tracking/process_station.py s3://my-bucket/UMI_Collection/station3/industrial_data -c example/tag_tracking/tag_tracking_station3.yaml -w ~/umi_data/station3/industrial_data --mcap --upload s3://my-bucket/mcaps/station3 --cleanup
```

With `--cleanup`, each task's local videos and MCAP files are deleted once its MCAP files are
verified on S3, so only one task's data (about 30 GB) is on disk at a time.

`--tasks task02,task03` for some of them; sessions already local: give the local folder
instead of the s3:// one (no `-w`). Disk: about 25 GB per task.

Upload the delivery folder:

```
aws s3 sync ~/umi_data/station3/delivery s3://my-bucket/deliveries/station3_2026-11-20
```

**A new station** (cameras, room or cubes not set up before) needs, once, between steps 3 and 4:
calibrated cameras (Camera calibration), a config (step 2 below:
`propose_station_config.py` on the first session), and the setup calibration on that session:

```
python run_tag_tracking_pipeline.py -c example/tag_tracking/tag_tracking_station3.yaml --stop_after segment_episodes ~/umi_data/station3/industrial_data/task01
python scripts_tag_tracking/calibrate_setup.py -i ~/umi_data/station3/industrial_data/task01 -c example/tag_tracking/tag_tracking_station3.yaml -o example/tag_tracking/setup_station3 -j 24
```

then `setup_calibration: setup_station3` in the config, and that session continues with
`-sf use_setup`:

```
python run_tag_tracking_pipeline.py -c example/tag_tracking/tag_tracking_station3.yaml -sf use_setup -nr ~/umi_data/station3/industrial_data/task01
```

## Calibration files

`example/tag_tracking/` holds the calibration of our setups; all paths in the configs are relative to it.

| File | Content |
|---|---|
| `tag_tracking.yaml` | setup of 2026-11-16 (task1-3): cameras by serial, bodies, episodes; format in `umi/tag_tracking/config.py` |
| `tag_tracking_station1.yaml` | station1 |
| `tag_tracking_station1_per_video.yaml` | station1, one episode per video with a fixed trim (test tasks) |
| `tag_tracking_station3.yaml` | station3 (same room and tags as 2026-11-16, other head and right wrist camera) |
| `aruco_config.yaml` | dictionary and tag sizes (black square edge, m) |
| `gripper_config*.yaml` | finger tags and TCP geometry per wrist camera; format in `umi/common/gripper_config_util.py` |
| `corner_noise.json` | tag corner noise (step 1) |
| `setup_20261116/`, `setup_station1/`, `setup_station3/` | setup calibration (step 2): tag map, cube shapes, cube -> wrist -> TCP |
| `delivery.yaml` | delivery declarations (vendor, dataset, task per session), template for step 6 |
| `openicc/<name>/cam/` | OpenICC intrinsics |
| `openicc/<name>/cam_imu/` | OpenICC camera-IMU calibration (fitted trajectory removed) |
| `*_setting_720*.yaml` | IMU noise per camera |
| `gopro_intrinsics_2_7k.json` | fallback intrinsics |

Cameras (GoPro HERO12, Wide, 2.7K 4:3, 59.94 fps, stabilization off). Calibrations belong to a
camera serial, not to a role: a camera can be head at one station and wrist at another.

| Serial | OpenICC | Used as |
|---|---|---|
| C3501326502664 | `openicc/head` | head (2026-11-16, station1) |
| C3501326029433 | `openicc/gripper2` | right wrist 2026-11-16, left wrist station1 |
| C3501326502891 | `openicc/gripper1` | left wrist (2026-11-16, station3) |
| C3501326500348 | `openicc/cam376` | right wrist station1 |
| C3501326495617 | `openicc/set2_head` | head station3 |
| C3501326501951 | `openicc/set2_left` | right wrist station3 (camera-IMU translation set to the HERO12 mean, see below) |

One config per setup: a new station, moved environment tags or a swapped gripper or camera
need their own config and setup calibration (step 2).

## Camera calibration

Once per camera (and after changing its lens or video settings): intrinsics (Kannala-Brandt
fisheye) and camera-IMU rotation / translation with OpenImuCameraCalibrator (OpenICC).

Board: ChArUco 12 x 9 squares, 30 mm squares, 22 mm markers, `DICT_5X5_100`, flat and rigid.

Record with the camera's take settings (2.7K 4:3, 59.94 fps, Wide, stabilization off), three videos:

| Video | How |
|---|---|
| `cam` | 30-60 s, move slowly around the board: all image regions, near and far, tilted views |
| `cam_imu` | 40-60 s, board in view, fast rotations about all three axes (turning, not only moving the camera), good light |
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
python scripts/prepare_openicc_dataset.py -o <calib_root> -n <name> --dry_run <video1.MP4> <video2.MP4> <video3.MP4>
python scripts/prepare_openicc_dataset.py -o <calib_root> -n <name> <video1.MP4> <video2.MP4> <video3.MP4>
```

Calibrate (15-25 min per camera; one terminal per camera to run several at once):

```
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v <calib_root>/<name>:/dataset \
    openicc:configurable-board python3 /OpenImuCameraCalibrator/python/run_gopro_calibration.py \
    --path_calib_dataset=/dataset --path_to_build=/OpenImuCameraCalibrator/build/applications \
    --num_squares_x=12 --num_squares_y=9 --checker_size_m=0.030 --marker_size_m=0.022 --aruco_dict=DICT_5X5_100 \
    --camera_model=FISHEYE --image_downsample_factor=1 --bias_calib_remove_s=2 \
    2>&1 | tee <calib_root>/<name>/calibration.log
```

Import into `example/tag_tracking/` (intrinsics, camera-IMU without the fitted trajectory, IMU
noise setting); prints the numbers to check and the config lines:

```
python scripts/import_openicc_calibration.py -i <calib_root>/<name> -n <name> -s <camera serial>
```

Check: intrinsics reprojection error about 0.3-1 px; camera-IMU translation about 20-25 mm
from zero (the IMU sits at the same place in every HERO12). The translation is only observable
with enough rotation in the `cam_imu` video; a value far from 20-25 mm (near 0, or far larger)
means it was not estimated. Record `cam_imu` again, or keep the rotation and time offset and set
the translation to the mean of our good HERO12 calibrations:

```
python scripts/import_openicc_calibration.py -i <calib_root>/<name> -n <name> -s <camera serial> --translation_mm=-0.2,-22.2,4.4
```

The estimate is kept in the json (`t_i_c_estimated`), and the delivery marks that IMU's
extrinsic `approximate`. Then add the camera to the tag tracking config (`cameras`,
`gripper_imu` or `head_imu`).

Print tags (black square `-t` mm, as in `aruco_config.yaml`):

```
python scripts/gen_tag_sticker_pdfs.py -o print/env_tags --ids 14-17,20-23,26-29,32-35,38-41,44-47 -t 80
```

## Recording

Per station, once:
- each gripper's cube has its own tags (two cubes with the same tag ids cannot be tracked);
- each camera keeps one role (head / left / right) for the whole day; label cameras with their serial.

Per task, in this order, nothing in between:
1. Gripper calibration: all cameras on, open and close each gripper fully 3-4 times (~15 s).
2. Head mapping: grippers resting, walk the head camera slowly past every environment tag (~30 s).
3. Takes, each at least 60 s: start all three cameras, then per episode: grippers on the dock,
   still for 1 s with the head camera looking at the dock tags; do the task; return to the dock.
   Stop all three cameras at the end. Check battery and card before each take.

Something changed (gripper bumped, tag re-stuck, table moved) or a clip went wrong: record a new
gripper calibration and head mapping, not a single clip. No short test clips between the head
mapping and the first take.

Upload per day: `<station>/<date>/head/`, `left/`, `right/`, files unchanged from the cameras.

## Session layout

```
<session>/raw_videos/*.MP4                      head and wrist videos of the take(s)
<session>/raw_videos/gripper_calibration/*.MP4  one open/close video per wrist camera
<session>/raw_videos/head_mapping/*.MP4         head camera walk past all environment tags
<session>/raw_videos/pairs.csv                  which videos were recorded together (from step 0)
```

## Run

`<config>` is the station's tag tracking config, e.g. `example/tag_tracking/tag_tracking_station3.yaml`;
`<session>` a session directory from step 0.

### 0. Sort a recording day into sessions

`scripts_tag_tracking/sort_recordings.py` on the day folder (`head/`, `left/`, `right/`), local
or `s3://` (aws CLI and its credentials; S3 videos are read through presigned URLs, nothing is
downloaded, sessions are server-side copies):

```
python scripts_tag_tracking/sort_recordings.py <day> --save <inventory.json> --propose <tasks.yaml>
python scripts_tag_tracking/sort_recordings.py <day> --tasks <tasks.yaml> -o <sessions>
```

The first command lists the recordings (videos of all cameras overlapping in time) and writes
the proposed tasks: from their tags, each short recording is a gripper calibration (a wrist's
fingers open and close fully), a head mapping (the head sees environment tags) or left out;
recordings of at least 60 s are takes; a task starts at the first take after a new head mapping.
Each short recording's role and evidence is listed in `tasks.yaml`; the second command copies
the videos of each task to `<sessions>/<task>/raw_videos/` with `pairs.csv`. Check `tasks.yaml`
for `unreadable` (video not readable after retries) and `left out` before the second command.
Tasks are named `task01`, `task02`, ...

### 1. Corner noise (once per camera model)

From a video of a static camera looking at static tags:

```
python scripts_tag_tracking/01_measure_corner_noise.py -i <static_video.MP4> -c <config>
```

### 2. Station setup (once per setup)

Draft the station's tag tracking and gripper configs from one sorted session:

```
python scripts_tag_tracking/propose_station_config.py <session> -n <station> --max_sec 600 \
    --head_intrinsics example/tag_tracking/openicc/<head>/cam/<cam_calib>.json [-t <config to copy settings from>]
```

It reads from the videos: camera serial per role (head / left / right from `pairs.csv`), finger
tags per wrist (from the gripper calibration video), cubes (cube-size tags the head sees moving
together), which cube is on which gripper (a wrist camera sees the other gripper's cube),
environment tags (head mapping) and dock tags (where both cubes rest, 3D, with the head
intrinsics). Writes `example/tag_tracking/tag_tracking_<station>.yaml` and
`gripper_config_<station>.yaml`, with the evidence in their header. Check it: cube-size tags never
seen are reported, not added; two cubes with the same tags show up as cubes split oddly.
The TCP is set to 192 mm in front of the lens and 75 mm below it (`--tcp_mm`,
`--cam_to_center_height`). Fill in `cameras`, `head_imu` and `gripper_imu` from the imports.

Setup calibration on one session of the station, with `setup_calibration` not set in the config:

```
python run_tag_tracking_pipeline.py -c <config> --stop_after segment_episodes <session>
python scripts_tag_tracking/check_wrist_events.py -i <session> -c <config>
python scripts_tag_tracking/calibrate_setup.py -i <session> -c <config> -o example/tag_tracking/setup_<station> -j 24
```

`check_wrist_events.py` plots the episode detection to `<session>/tag_tracking/wrist_events/`;
tune the `episodes` section if episodes are missed. A recording with a video the audio sync
rejects (`failed` in `logs/04_audio_sync.log`) is left out whole, with a warning
(`left out, a video of the recording has no audio sync offset`): its episodes cannot be synced.

Check `setup_<station>/report.json`: `cube_to_wrist` spread of a few mm and most estimates
inliers; for a station in a room mapped before, the tag map should agree with the earlier one
within a few mm. Then set in the config:

```yaml
setup_calibration: setup_<station>
```

After changing only the TCP geometry in the gripper config:

```
python scripts_tag_tracking/calibrate_setup.py -c <config> -o example/tag_tracking/setup_<station> --tcp_only
```

### 3. Sessions

```
python run_tag_tracking_pipeline.py -c <config> -j 16 -nr <session>
python scripts_tag_tracking/07_export_mcap.py -i <session> -j 4
```

The session used for the setup calibration continues with `-sf use_setup`. The end of the
`export_plan` log gives the yield: episodes, seconds used, and why episodes were dropped
(tracking gap, a video stopped during it, cubes never left the dock).

| Option | |
|---|---|
| `-j` | parallel processes, default all cores |
| `-nr` | skip the render step |
| `--start_from <step>` / `--stop_after <step>` | run a range of steps |

Steps: `sort_videos imu wrist_tags gripper_range audio_sync head_tags segment_episodes use_setup
calibrate_bodies track_bodies fuse_head_imu trim_docking gripper_tcp sync_wrists fuse_grippers render export_plan`.
Logs: `<session>/logs/`.

All sessions of a station in one run (`scripts_tag_tracking/process_station.py`):

```
python scripts_tag_tracking/process_station.py <sessions> -c <config> [-w <local folder>] [--tasks task02,task03] [--mcap] [--upload s3://...] [--cleanup]
```

Per session, in order: download (`<sessions>` on `s3://`: to `-w`, files already there skipped),
the pipeline (all steps, no render), the dataset plan again with `--keep_gaps` (unless
`--no_keep_gaps`), with `--mcap` the MCAP files (`<session>/dataset_mcap/`; files of an earlier
export are deleted first; stops if the free disk is under 1.2 x the session's videos), with
`--upload` the MCAP files to
`<upload>/<session>/`. A failing session does not stop the others. At the end a summary per
session: status, the failed step and its log, episodes and seconds used, MCAP files. Each
session's output is also in `<session>/logs/process_station.log`. Without `--tasks`: every
sub-folder with `raw_videos/`. The station's setup calibration must exist (step 2).
`--start_from pipeline|plan|mcap|upload` resumes a session after a failure (the outputs of the
earlier steps must be there), e.g. `--tasks task05 --start_from mcap` after freeing disk.
Disk: a session on S3 is downloaded only if the free disk holds it plus `--spare_gb` (5 GB:
processing outputs, one MCAP batch); otherwise it is skipped before downloading (summary:
`skipped`, enlarge the disk). Session sizes: 10-25 GB (station3 task01: 23.5 GB), so give the
instance at least the largest session + 10 GB free.
With `--upload`, the MCAP files are exported, uploaded, checked on S3 and deleted locally
`--batch_size` (10) episodes at a time, so a session needs about its videos plus one batch
(task of 10 min: about 12-15 GB). Without `--upload` all MCAP files stay: about 2 x the videos.

Several stations on one machine, one after the other, one command: list them in a job file
(`example/tag_tracking/process_jobs.yaml`: per station `sessions`, `config`, `workdir`,
optional `tasks`, `upload`) and pass the common options:

```
python scripts_tag_tracking/process_station.py --jobs example/tag_tracking/process_jobs.yaml --mcap --cleanup
```

All stations are checked before the first starts (configs exist, tasks found); one summary
for all at the end.

`--cleanup` (limited disk; sessions on `s3://` and `--upload` only): after a session's upload,
if every local MCAP file is on S3 with the same size, its local videos (`demos/*/raw_video.mp4`,
`raw_videos/`) and MCAP files are deleted, about 30 GB per task. Kept: logs, `pairs.csv`,
tag tracking outputs, `dataset_plan.pkl`. The videos stay in the S3 session; running the session
again downloads them again. Nothing is deleted if a file is missing or differs on S3.

After a TCP change, without re-tracking:

```
python scripts_tag_tracking/00c_use_setup.py -i <session> -c <config>
python scripts_tag_tracking/06_export_dataset_plan.py -i <session> -c <config>
python scripts_tag_tracking/07_export_mcap.py -i <session> -j 4
```

### 4. Checks

How many episodes and how long, over a folder of MCAP files (local or `s3://`, only the
metadata at the end of each file is read): per task folder and in total, good episodes (a pose
for every step of both grippers) and episodes kept with tracking gaps:

```
python scripts_tag_tracking/report_episodes.py <folder or s3://...> [--csv episodes.csv]
```

What happens in each MCAP, one CSV row per file: 6 head camera frames per MCAP (decoded from
the MCAP, ranged reads on S3) described by Claude: task sentence, objects, hands used, visible
outcome, notes. Rows already in the CSV are skipped (a stopped run continues); `--sheets <dir>`
saves the frames Claude saw; `--dry_run` makes only the sheets. Needs Anthropic credentials.

```
python scripts_tag_tracking/describe_mcaps.py <folder or s3://...> -o mcap_tasks.csv [--sheets <dir>]
```

One line per task (station, day, task folder) from those descriptions, and the episodes that do
not fit it (description of another task, outcome not success, notes):

```
python scripts_tag_tracking/summarize_tasks.py mcap_tasks.csv -o task_summary
```

Writes `task_summary_tasks.csv` (station, day, task, episodes, minutes, good episodes, task line,
objects) and `task_summary_mismatches.csv` (one row per flagged episode with the reason).

```
python scripts_tag_tracking/check_tcp_frame.py -i <session> -c <config>
python scripts_tag_tracking/check_tcp_frame.py -i <session> -c <config> --random 10
python scripts_tag_tracking/04_render_tracking.py -i <session> -c <config>
```

`check_tcp_frame.py`: the head frame with each gripper's TCP (green), a line down to the table
(yellow) and the cube axes, and each gripper's wrist camera frame at the same instant with its
TCP; at the plan step with the lowest TCP. `--random 10` (`--seed`) or `-e 3,7`: one image per
plan episode, at that episode's lowest TCP. Images: `<session>/tag_tracking/tcp_check_*.png`.

Frame sync of the exported videos (see Frame sync check below):

```
python scripts_tag_tracking/check_frame_sync.py -i <session> -c <config>
```

### 5. Training dataset

```
python scripts_slam_pipeline/07_generate_replay_buffer.py --no_mirror -o <dataset.zarr.zip> <session>
```

### 6. Odyssey delivery

Odyssey Robotics Data Capture Spec v0.6, container `mcap`. Needs `pip install pyarrow`.

Input:
- a processed session with `dataset_plan.pkl` (step 3). For sessions with tracking gaps, export
  the plan with `--keep_gaps` first: episodes stay whole and frames without a pose are marked
  instead of the episode being dropped;
- the session's tag tracking config;
- a copy of `example/tag_tracking/delivery.yaml` filled in: `delivery:` (vendor, dataset, delivery
  id, rig, license, anonymization) and one entry per session directory name (`task_text`,
  `task_family`, ...). Anything left null is exported as null.

```
python scripts_tag_tracking/06_export_dataset_plan.py -i <session> -c <config> --keep_gaps
python scripts_tag_tracking/13_export_delivery.py -i <session> -c <config> -d <delivery.yaml> -o <delivery_dir> -j 16
```

`-e 0,3,6` exports only these plan episodes. Several sessions can go to the same `-o`: the
manifest keeps the episodes of earlier runs and replaces re-exported ones.

Output, one directory per episode:

| Path | |
|---|---|
| `<delivery_dir>/<episode_id>/episode.mcap` | topics as in MCAP below, message times in UTC |
| `<delivery_dir>/<episode_id>/episode.json` | ids, UTC capture start / end, frame rate, `clock_sync_accuracy_ns`, world frame, gripper range, task |
| `<delivery_dir>/<episode_id>/sensors.json` | per camera, IMU, TCP pose, gripper: MCAP topic, schema, units, clock offset, intrinsics, extrinsics |
| `<delivery_dir>/<episode_id>/frames.parquet` | one row per head camera frame in `episode.mcap` (from the keyframe before the task, so `camera_frame_index` starts at 0): UTC timestamp, TCP poses (xyz + quaternion xyzw), `pose_valid`, gripper widths, head camera pose |
| `<delivery_dir>/<episode_id>/annotations.json` | event timeline (spec 2.7) from the session's annotations (08 / 09), times from the episode's first frame; only where the session is annotated |
| `<delivery_dir>/manifest.parquet` | one row per episode with every file's size and sha256, written last |

`episode_id` = head camera serial + UTC start of the head video + head frame of the task
start; the same on every re-export. `episode.json` `task_interval_s`: the task (bang to bang)
in seconds from the first frame; the frames before it are the lead-in to the keyframe.

Annotations: run before the export, per session (needs `ANTHROPIC_API_KEY`):

```
python scripts_tag_tracking/08_annotate_episodes.py -i <session> -c <config>
python scripts_tag_tracking/09_normalize_labels.py -i <session>
```

The export moves each annotation onto the delivered episode and takes the task text and family
(where the delivery yaml declares none), outcome, success and failure spans from it. Without
annotations, or with `--no_annotations`: no `annotations.json`; task text, family and outcome as
declared in the delivery yaml (`outcome` `unknown` where not declared). `validation_status` is
`not_run` (no quality checks).

### 7. Rhoda delivery (three files per recording)

The data PRD's UMI delivery: one folder per recording (episode) with the three camera videos, the
per-frame data and the metadata. The videos are not re-encoded.

```
python scripts_tag_tracking/14_export_rhoda.py -i <session> -c <config> -o <out> --name <station>_<date>_<task> [--mosaic 3]
```

| File | Content |
|---|---|
| `<rec>_head.mp4`, `_left.mp4`, `_right.mp4` | the cameras' original H.265, copied from the keyframe at or before the episode's first step to its last step (GoPro: one keyframe per second, so up to 1 s before the episode); tagged `hvc1` for QuickTime / Safari |
| `<rec>_frames.csv` | one row per step: the frame index of the step in each file, time from the first step, per gripper (`right` = robot0, `left` = robot1) width in m (0 = fully closed), `pose_valid` and TCP pose in the world frame (m, quaternion xyzw), empty where `pose_valid` is 0 |
| `<rec>_meta.json` | per camera model, lens, field of view, resolution, frame rate, Kannala-Brandt intrinsics and their source; sync method, sync event and `sync_frame_index` per file, wrist clock corrections and uncertainty; gripper width and TCP conventions |
| `<rec>_mosaic.mp4` | optional review video (`--mosaic N` episodes evenly over the session, or `--mosaic_episodes`): head on top, right and left wrist below, from the sync frames on, H.264 yuv420p 30 fps; the only part that decodes and encodes (`--mosaic_encoder h264_nvenc` on a GPU) |

The sync event is the episode's first step, the same instant in all three files (each camera's frame
nearest to it, within half a frame): frame f of a file belongs to step `f - sync_frame_index`.

## Frame sync check

`check_frame_sync.py` checks the exported frame timestamps against the cameras' own images,
without the IMUs and without the motion sync it is checking.

Input: a processed session with `dataset_plan.pkl` and its tag tracking config. Uses per
episode the wrist videos' tag detections (`demos/*/tag_detection.pkl`), the head tracking
(`demos/<head>/tag_tracking.pkl`), `tag_tracking/bodies.json` and `gripper_tcp.json`.

```
python scripts_tag_tracking/check_frame_sync.py -i <session> -c <config> [-e 0,3,6] [--min_tags 1] [-j 16]
```

Per episode and wrist camera, the wrist camera's orientation from the environment tags in its
own video (at the wrist frame timestamps of the export) is compared with the same camera's
orientation from the head video (cube pose x cube-to-wrist, at the head frame times); the time
shift that matches them best is the frame sync error.

Output (printed, nothing written):
- per episode and wrist camera: frames used, rotation speed, video sync error to the head
  camera (ms and frames), the mismatch at the best shift, at 0 and at +-1 frame (how well the
  episode can time a one-frame error), and the wrist IMU against the wrist video (ms);
- per episode with both wrists timed: the three pairs head-right, head-left, right-left;
- per wrist video: median and 5-95 % over the episodes that can time a one-frame error, and
  the right-left median.

0 ms = in sync; one frame is 16.7 ms. A positive error means the wrist frames are stamped too
early relative to the head video. Sync means equal timestamps show the same instant; each
camera's video in the MCAP starts at its own keyframe up to 1 s before the episode, so match
frames by timestamp, not by frame number.

To check the numbers: the configs and calibrations are in `example/tag_tracking/` (table
above); the processed session directories are not in the repository.

## Outputs

| Path | |
|---|---|
| `<session>/tag_tracking/episodes.json` | episodes per head video (head video seconds), task interval without docking |
| `<session>/tag_tracking/wrist_sync.json` | wrist clock corrections and sync accuracy |
| `<session>/demos/demo_<head>/tag_tracking.pkl` | head, cube and fused gripper poses |
| `<session>/dataset_plan.pkl` | episodes in the UMI dataset plan format |
| `<session>/dataset_mcap/episode_*.mcap` | one file per episode (07) |
| `<delivery_dir>/...` | Odyssey delivery (step 6) |
| `<session>/tag_tracking/tcp_check_*.png` | TCP check renders |

## MCAP

| Topic | Schema |
|---|---|
| `robot{i}/eef/state` | `umi.msg.EefState`: TCP `pos` (m), `rot_axis_angle` (rad), world frame |
| `robot{i}/eef/pose` | `umi.msg.PoseQuat`: the same TCP poses, `pos` (m) + `quat_xyzw` |
| `robot{i}/gripper/state` | `umi.msg.GripperState`: `width` (m) |
| `robot{i}/eef_demo_start/state`, `robot{i}/eef_demo_end/state` | first / last TCP pose of the episode |
| `robot{i}/imu` | `umi.msg.ImuSample`: wrist camera IMU |
| `/cameras/camera{j}/video` | `foxglove.CompressedVideo`, H.265 from the GoPro files, not re-encoded |
| `/head/pose` | `umi.msg.PoseQuat`: head camera pose in the world per tracked head frame (with `-c`) |
| `/head/imu` | `umi.msg.ImuSample`: head camera IMU (with `-c` and `head_imu` in the config) |

`robot0` right, `robot1` left; `camera0` robot0 wrist, `camera1` robot1 wrist, `camera2` head.
No pose message where the pose is not available (tracking gap, plan with `--keep_gaps`);
the count per robot is in metadata `recording` (`robot{i}_pose_missing_steps`).
Metadata `recording`: episode length, image size, fps, world frame tag, TCP in the wrist camera frame.
With `-c`: metadata `camera{j}` (model, lens, field of view, resolution, frame rate,
Kannala-Brandt intrinsics) and `sync` (method, per camera the frame of the episode's first
step in the original video and its clock correction).

## Limitations

- Episodes with a tracking gap longer than `max_gap_sec` (0.25 s) inside are not exported to
  `dataset_plan.pkl`, unless `--keep_gaps` (then not usable for the replay buffer).
- Each camera's video in an MCAP starts at its own keyframe before the episode (not re-encoded),
  so the three videos start up to 1 s apart.
- A task ends when the first gripper returns to the dock.
