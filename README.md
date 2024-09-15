# Universal Manipulation Interface

[[Project page]](https://umi-gripper.github.io/)
[[Paper]](https://umi-gripper.github.io/#paper)
[[Hardware Guide]](https://docs.google.com/document/d/1TPYwV9sNVPAi0ZlAupDMkXZ4CA1hsZx7YDMSmcEy6EU/edit?usp=sharing)
[[Data Collection Instruction]](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Tutorial-4db1a1f0f2aa4a2e84d9742720428b4c?pvs=4)
[[SLAM repo]](https://github.com/cheng-chi/ORB_SLAM3)
[[SLAM docker]](https://hub.docker.com/r/chicheng/orb_slam3)

<img width="90%" src="assets/umi_teaser.png">

[Cheng Chi](http://cheng-chi.github.io/)<sup>1,2</sup>,
[Zhenjia Xu](https://www.zhenjiaxu.com/)<sup>1,2</sup>,
[Chuer Pan](https://chuerpan.com/)<sup>1</sup>,
[Eric Cousineau](https://www.eacousineau.com/)<sup>3</sup>,
[Benjamin Burchfiel](http://www.benburchfiel.com/)<sup>3</sup>,
[Siyuan Feng](https://www.cs.cmu.edu/~sfeng/)<sup>3</sup>,

[Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)<sup>3</sup>,
[Shuran Song](https://www.cs.columbia.edu/~shurans/)<sup>1,2</sup>

<sup>1</sup>Stanford University,
<sup>2</sup>Columbia University,
<sup>3</sup>Toyota Research Institute

## 🛠️ Installation
Only tested on Ubuntu 22.04

Install docker following the [official documentation](https://docs.docker.com/engine/install/ubuntu/) and finish [linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall/).

Install system-level dependencies:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```

Activate environment
```console
$ conda activate umi
(umi)$ 
```

## Running UMI SLAM pipeline
Download example data
```console
(umi)$ wget --recursive --no-parent --no-host-directories --cut-dirs=2 --relative --reject="index.html*" https://real.stanford.edu/umi/data/example_demo_session/
```

Run SLAM pipeline
```console
(umi)$ python run_slam_pipeline.py example_demo_session

...
Found following cameras:
camera_serial
C3441328164125    5
Name: count, dtype: int64
Assigned camera_idx: right=0; left=1; non_gripper=2,3...
             camera_serial  gripper_hw_idx                                     example_vid
camera_idx                                                                                
0           C3441328164125               0  demo_C3441328164125_2024.01.10_10.57.34.882133
99% of raw data are used.
defaultdict(<function main.<locals>.<lambda> at 0x7f471feb2310>, {})
n_dropped_demos 0
````
For this dataset, 99% of the data are useable (successful SLAM), with 0 demonstrations dropped. If your dataset has a low SLAM success rate, double check if you carefully followed our [data collection instruction](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Instruction-4db1a1f0f2aa4a2e84d9742720428b4c). 

Despite our significant effort on robustness improvement, OBR_SLAM3 is still the most fragile part of UMI pipeline. If you are an expert in SLAM, please consider contributing to our fork of [OBR_SLAM3](https://github.com/cheng-chi/ORB_SLAM3) which is specifically optimized for UMI workflow.

Generate dataset for training.
```console
(umi)$ python scripts_slam_pipeline/07_generate_replay_buffer.py -o example_demo_session/dataset.zarr.zip example_demo_session
```

## Training Diffusion Policy
Single-GPU training. Tested to work on RTX3090 24GB.
```console
(umi)$ python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=example_demo_session/dataset.zarr.zip
```

Multi-GPU training.
```console
(umi)$ accelerate --num_processes <ngpus> train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=example_demo_session/dataset.zarr.zip
```

Downloading in-the-wild cup arrangement dataset (processed).
```console
(umi)$ wget https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip
```

Multi-GPU training.
```console
(umi)$ accelerate --num_processes <ngpus> train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=cup_in_the_wild.zarr.zip
```

## 🦾 Real-world Deployment
In this section, we will demonstrate our real-world deployment/evaluation system with the cup arrangement policy. While this policy setup only requires a single arm and camera, the our system supports up to 2 arms and unlimited number of cameras.

### ⚙️ Hardware Setup
1. Build deployment hardware according to our [Hardware Guide](https://docs.google.com/document/d/1TPYwV9sNVPAi0ZlAupDMkXZ4CA1hsZx7YDMSmcEy6EU).
2. Setup UR5 with teach pendant:
    * Obtain IP address and update [eval_robots_config.yaml](example/eval_robots_config.yaml)/robots/robot_ip.
    * In Installation > Payload
        * Set mass to 1.81 kg
        * Set center of gravity to (2, -6, 37)mm, CX/CY/CZ.
    * TCP will be set automatically by the eval script.
    * On UR5e, switch control mode to remote.

    If you are using Franka, follow this [instruction](franka_instruction.md).
3. Setup WSG50 gripper with web interface:
    * Obtain IP address and update [eval_robots_config.yaml](example/eval_robots_config.yaml)/grippers/gripper_ip.
    * In Settings > Command Interface
        * Disable "Use text based Interface"
        * Enable CRC
    * In Scripting > File Manager
        * Upload [umi/real_world/cmd_measure.lua](umi/real_world/cmd_measure.lua)
    * In Settings > System
        * Enable Startup Script
        * Select `/user/cmd_measure.lua` you just uploaded.
4. Setup GoPro:
    * Install GoPro Labs [firmware](https://gopro.com/en/us/info/gopro-labs).
    * Set date and time.
    * Scan the following QR code for clean HDMI output 
    <br><img width="50%" src="assets/QR-MHDMI1mV0r27Tp60fWe0hS0sLcFg1dV.png">
5. Setup [3Dconnexion SpaceMouse](https://www.amazon.com/3Dconnexion-SpaceMouse-Wireless-universal-receiver/dp/B079V367MM):
    * Install libspnav `sudo apt install libspnav-dev spacenavd`
    * Start spnavd `sudo systemctl start spacenavd`

### 🤗 Reproducing the Cup Arrangement Policy ☕
Our in-the-wild cup arragement policy is trained with the distribution of ["espresso cup with saucer"](https://www.amazon.com/s?k=espresso+cup+with+saucer) on Amazon across 30 different locations around Stanford. We created a [Amazon shopping list](https://www.amazon.com/hz/wishlist/ls/Q0T8U2N5U3IU?ref_=wl_share) for all cups used for training. We published the processed [Zarr dataset and](https://real.stanford.edu/umi/data/zarr_datasets) pre-trained [checkpoint](https://real.stanford.edu/umi/data/pretrained_models/) (finetuned CLIP ViT-L backbone).

<img width="90%" src="assets/umi_cup.gif">

Download pre-trained checkpoint.
```console
(umi)$ wget https://real.stanford.edu/umi/data/pretrained_models/cup_wild_vit_l_1img.ckpt
```

Grant permission to the HDMI capture card.
```console
(umi)$ sudo chmod -R 777 /dev/bus/usb
```

Launch eval script.
```console
(umi)$ python eval_real.py --robot_config=example/eval_robots_config.yaml -i cup_wild_vit_l.ckpt -o data/eval_cup_wild_example
```
After the script started, use your spacemouse to control the robot and the gripper (spacemouse buttons). Press `C` to start the policy. Press `S` to stop.

If everything are setup correctly, your robot should be able to rotate the cup and placing it onto the saucer, anywhere 🎉

Known issue ⚠️: The policy doesn't work well under direct sunlight, since the dataset was collected during a rainiy week at Stanford.

## 📚 Dataset Format
UMI has multiple tiers of data storage formats:
* GoPro data: Just a folder of GoPro mp4s :)
* SLAM data: Output of ORB_SLAM3 pipeline (volatile)
* Zarr data: A single zip file optimized for fast random read for training.

### Zarr data format
Following [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), UMI uses [Zarr](https://zarr.dev/) as the container for training datasets. Zarr is similar to [HDF5](https://docs.hdfgroup.org/hdf5/v1_14/_intro_h_d_f5.html) but offers better flexibility for storage backends, chunking, compressors and parallel access. 

Conceptually, Zarr can be understood as a nested `dict` of "numpy arrays". For example, here's the structure of the `example_demo_session` dataset.
``` python
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

root = zarr.open('example_demo_session/dataset.zarr.zip')
print(root.tree())
>>>
/
 ├── data
 │   ├── camera0_rgb (2315, 224, 224, 3) uint8
 │   ├── robot0_demo_end_pose (2315, 6) float64
 │   ├── robot0_demo_start_pose (2315, 6) float64
 │   ├── robot0_eef_pos (2315, 3) float32
 │   ├── robot0_eef_rot_axis_angle (2315, 3) float32
 │   └── robot0_gripper_width (2315, 1) float32
 └── meta
     └── episode_ends (5,) int64
```


#### ReplayBuffer
We implemented `ReplayBuffer` class for convenience of accessing zarr data.
```python
from diffusion_policy.common.replay_buffer import ReplayBuffer

replay_buffer = ReplayBuffer.create_from_group(root)
replay_buffer.n_episodes
>>> 5

# reading an episode
ep = replay_buffer.get_episode(0)
ep.keys()
>>> dict_keys(['camera0_rgb', 'robot0_demo_end_pose', 'robot0_demo_start_pose', 'robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width'])

ep['robot0_gripper_width']
>>>
array([[0.07733118],
       [0.07733118],
       [0.07734068],
...
       [0.08239228],
       [0.08236252],
       [0.0823558 ]], dtype=float32)
```


#### Data Group
In `root['data']` "dict", we have a group of arrays containing demonstration episodes, concatinated along the first dimension (time/step). In this dataset, we have a total of 2315 steps across 5 episodes. In UMI, we assume data has a frame rate of 60Hz (actually, 59.94Hz), matching the recording frame rate of GoPros. All arrays in `root['data']` must have the same size in their first (time) dimension.

```python
root['data']['robot0_eef_pos']
>>> <zarr.core.Array '/data/robot0_eef_pos' (2315, 3) float32>

root['data']['robot0_eef_pos'][0]
>>> array([ 0.1872826 , -0.35130176,  0.1859438 ], dtype=float32)

root['data']['robot0_eef_pos'][:]
>>>
array([[ 0.1872826 , -0.35130176,  0.1859438 ],
       [ 0.18733297, -0.3509169 ,  0.18603411],
       [ 0.18735182, -0.3503186 ,  0.18618457],
       ...,
       [ 0.12694108, -0.3326249 ,  0.13230264],
       [ 0.12649481, -0.3347473 ,  0.1347403 ],
       [ 0.12601827, -0.33651358,  0.13699797]], dtype=float32)

```
#### Metadata Group
How do we know the start and end of each episode? We store an integer array `root['meta']['episode_ends']` that contains the `end` index of each episode into `data` arrays. 
For example, the first episode can be accessed with `root['data']['robot0_eef_pos'][0:468]` and the second episode can be accessed with `root['data']['robot0_eef_pos'][468:932]`.

```python
root['meta']['episode_ends'][:]
>>> array([ 468,  932, 1302, 1710, 2315])
```


#### Data Array Chunking and Compression
Note that all arrays in the dataset are of type `zarr.core.Array` instead of `numpy.ndarray`. While offerring similar API to numpy arrays, Zarr arrays are optimized for fast on-disk storage with *chunked compression*. For example, camera images `root['data']['camera0_rgb']` is stored with chunk size `(1, 224, 224, 3)` and `JpegXl` compression. When reading from a zarr array, an entire chunk of data is loaded from disk storage and de-compressed to a numpy array.

For optimal performance, you want to carefully chose your chunk size. A chunk size too big means that you are de-compressing more data than necessary (e.g. chunks=(100, 224, 244, 3) will decompress and discard 99 images when accessing [0]). In contrast, having the chunk size too small will incur additional overhead and reduces compression rate (.e.g chunks=(1,14,14,3) means each image is split into 256 chunks).

```python
root['data']['camera0_rgb']
>>> <zarr.core.Array '/data/camera0_rgb' (2315, 224, 224, 3) uint8>

root['data']['camera0_rgb'].chunks # chunk size
>>> (1, 224, 224, 3)

root['data']['camera0_rgb'].nchunks # number of chunks
>>> 2315

root['data']['camera0_rgb'].compressor
>>> JpegXl(decodingspeed=None, distance=None, effort=None, index=None, keeporientation=None, level=99, lossless=False, numthreads=1, photometric=None, planar=None, usecontainer=None)

root['data']['camera0_rgb'][0]
>>>
array([[[ 7,  6, 15],
        [ 7,  6, 15],
        [ 4,  4, 13],
        ...,
        [ 6,  7, 15],
        [ 4,  7, 14],
        [ 3,  6, 13]],

       ...,

       [[ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        ...,
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0]]], dtype=uint8)
```

We use a rather large chunk size for low-dimisional data. Since these data are cached into numpy array inside `UmiDataset`, no read IOPS overhead is introduced.

``` python
root['data']['robot0_eef_pos'].chunks
>>> (468, 3)

root['data']['robot0_eef_pos'].compressor # uncompressed chunks
>>> None
```

#### In-memory Compression
During traning, streaming dataset from a network drive is often bottelnecked by [IOPS](https://en.wikipedia.org/wiki/IOPS), especially when multiple GPUs/nodes reading from the same network drive. While loading the entire dataset to memory works around IOPS bottleneck, an uncompressed UMI dataset often don't fit in RAM.

We found streaming *compressed* dataset from RAM to be a good tradeoff between memory footprint and read performance.
```python
root.store
>>> <zarr.storage.ZipStore at 0x76b73017d400>

ram_store = zarr.MemoryStore()
# load stored chunks in bytes directly to memory, without decompression
zarr.convenience.copy_store(root.store, ram_store)
ram_root = zarr.group(ram_store)
print(ram_root.tree())
>>>
/
 ├── data
 │   ├── camera0_rgb (2315, 224, 224, 3) uint8
 │   ├── robot0_demo_end_pose (2315, 6) float64
 │   ├── robot0_demo_start_pose (2315, 6) float64
 │   ├── robot0_eef_pos (2315, 3) float32
 │   ├── robot0_eef_rot_axis_angle (2315, 3) float32
 │   └── robot0_gripper_width (2315, 1) float32
 └── meta
     └── episode_ends (5,) int64


# loading compressed data to RAM with ReplayBuffer
ram_replay_buffer = ReplayBuffer.copy_from_store(
    root.store,
    zarr.MemoryStore()
)
ep = ram_replay_buffer.get_episode(0)
```


## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## 🙏 Acknowledgement
* Our GoPro SLAM pipeline is adapted from [Steffen Urban](https://github.com/urbste)'s [fork](https://github.com/urbste/ORB_SLAM3) of [OBR_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3).
* We used [Steffen Urban](https://github.com/urbste)'s [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator/) for camera and IMU calibration.
* The UMI gripper's core mechanism is adpated from [Push/Pull Gripper](https://www.thingiverse.com/thing:2204113) by [John Mulac](https://www.thingiverse.com/3dprintingworld/designs).
* UMI's soft finger is adapted from [Alex Alspach](http://alexalspach.com/)'s original design at TRI.
