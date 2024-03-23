#!/bin/sh

conda init
conda activate umi
cd /ws/universal_manipulation_interface
python run_slam_pipeline.py example_demo_session
python scripts_slam_pipeline/07_generate_replay_buffer.py -o example_demo_session/dataset.zarr.zip example_demo_session
python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=example_demo_session/dataset.zarr.zip
