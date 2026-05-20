"""Offline prediction on the grasp-cube UMI training dataset."""

# ===== Imports =====
import os
import pathlib
import sys
import json

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_umi_action,
    get_real_umi_obs_dict,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)
register_codecs()

# ===== Variables =====
CKPT_PATH = "data/outputs/2026.04.27/05.26.56_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt"
OUTPUT_DIR = "test_inference/results/grasp_cube_predictions"
DEVICE = "cuda:0"
EPISODE_ID = 0
HORIZON_INDEX = 15

# ===== Setup =====
def load_policy(ckpt_path, device):
    payload = torch.load(
        open(ckpt_path, "rb"),
        map_location="cpu",
        pickle_module=dill,
    )
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(
        payload,
        exclude_keys=None,
        include_keys=None,
    )

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()

    return policy, cfg

def load_dataset(cfg):
    return hydra.utils.instantiate(cfg.task.dataset)

# ===== Prediction Functions =====
def get_demo_info(dataset, episode_id):
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:])
    if episode_id < 0 or episode_id >= len(episode_ends):
        raise ValueError(
            f"episode_id={episode_id} is outside dataset with "
            f"{len(episode_ends)} episodes."
        )

    start_idx = 0 if episode_id == 0 else int(episode_ends[episode_id - 1])
    end_idx = int(episode_ends[episode_id])
    start_pos = np.asarray(dataset.replay_buffer["robot0_eef_pos"][start_idx])
    start_rot = np.asarray(dataset.replay_buffer["robot0_eef_rot_axis_angle"][start_idx])
    demo_start_pose = np.concatenate([start_pos, start_rot], axis=-1)
    return start_idx, end_idx, demo_start_pose


def get_ground_truth_state(dataset, idx):
    return {
        "eef_pos": np.asarray(dataset.replay_buffer["robot0_eef_pos"][idx]),
        "eef_rot_axis_angle": np.asarray(
            dataset.replay_buffer["robot0_eef_rot_axis_angle"][idx]
        ),
        "gripper_width": np.asarray(dataset.replay_buffer["robot0_gripper_width"][idx]),
    }


def build_rollout_inputs(dataset, current_idx, start_idx, rollout_state, cfg, demo_start_pose):
    # References: BimanualUmiEnv.get_obs defines the raw env_obs keys, and
    # SequenceSampler.sample_sequence pads early observation windows this way.
    def obs_window_indices(horizon, down_sample_steps):
        num_valid = min(horizon, (current_idx - start_idx) // down_sample_steps + 1)
        first_idx = current_idx - (num_valid - 1) * down_sample_steps
        indices = np.arange(first_idx, current_idx + 1, down_sample_steps)
        if len(indices) < horizon:
            pad = np.full(horizon - len(indices), indices[0], dtype=indices.dtype)
            indices = np.concatenate([pad, indices], axis=0)
        return indices

    def state_at(idx):
        if idx in rollout_state:
            return rollout_state[idx]
        return get_ground_truth_state(dataset, idx)

    env_obs = {}
    timestamp_indices = np.asarray([current_idx], dtype=np.int64)

    for key, attr in cfg.task.shape_meta.obs.items():
        obs_type = attr.get("type", "low_dim")
        horizon = int(attr["horizon"])
        down_sample_steps = int(attr["down_sample_steps"])
        indices = obs_window_indices(horizon, down_sample_steps)
        timestamp_indices = indices

        if obs_type == "rgb":
            env_obs[key] = np.asarray(dataset.replay_buffer[key][indices])
        elif key == "robot0_eef_pos":
            env_obs[key] = np.stack([
                state_at(int(idx))["eef_pos"]
                for idx in indices
            ]).astype(np.float32)
        elif key == "robot0_eef_rot_axis_angle":
            env_obs[key] = np.stack([
                state_at(int(idx))["eef_rot_axis_angle"]
                for idx in indices
            ]).astype(np.float32)
        elif key == "robot0_gripper_width":
            env_obs[key] = np.stack([
                state_at(int(idx))["gripper_width"]
                for idx in indices
            ]).astype(np.float32)

    env_obs["timestamp"] = timestamp_indices.astype(np.float64)

    # Reference: eval_real.py calls get_real_umi_obs_dict right before policy inference.
    model_obs = get_real_umi_obs_dict(
        env_obs=env_obs,
        shape_meta=cfg.task.shape_meta,
        obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
        tx_robot1_robot0=None,
        episode_start_pose=[demo_start_pose],
    )
    return env_obs, model_obs


def predict_absolute_horizon(policy, model_obs, env_obs, cfg, device):
    # Reference: eval_real.py does this same policy call, then converts
    # result["action_pred"] with get_real_umi_action.
    device = torch.device(device)
    obs = dict_apply(
        model_obs,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
    )
    with torch.no_grad():
        result = policy.predict_action(obs)
    raw_action_horizon = result["action_pred"][0].detach().to("cpu").numpy()
    abs_action_horizon = get_real_umi_action(
        raw_action_horizon,
        env_obs,
        cfg.task.pose_repr.action_pose_repr,
    )
    return raw_action_horizon, abs_action_horizon


def advance_rollout(dataset, rollout_state, current_idx, end_idx, abs_action_horizon, cfg, horizon_index):
    if horizon_index <= 0:
        raise ValueError("horizon_index must be > 0 so the rollout advances.")

    step = int(cfg.task.shape_meta.action.down_sample_steps)
    remaining_steps = end_idx - 1 - current_idx
    if remaining_steps <= 0:
        state = get_ground_truth_state(dataset, current_idx)
        return current_idx, 0, state, state

    used_horizon_index = min(
        horizon_index,
        max(1, int(np.ceil(remaining_steps / step))),
    )
    matched_idx = min(current_idx + used_horizon_index * step, end_idx - 1)

    # Store every predicted point because the next observation window can look
    # backward into earlier predicted timesteps.
    for horizon_idx, action in enumerate(abs_action_horizon):
        timestep_idx = current_idx + horizon_idx * step
        if timestep_idx >= end_idx:
            break

        rollout_state[int(timestep_idx)] = {
            "eef_pos": np.asarray(action[:3], dtype=np.float32),
            "eef_rot_axis_angle": np.asarray(action[3:6], dtype=np.float32),
            "gripper_width": np.asarray(action[6:7], dtype=np.float32),
        }

    matched_action = abs_action_horizon[used_horizon_index]
    pred_state = {
        "eef_pos": matched_action[:3],
        "eef_rot_axis_angle": matched_action[3:6],
        "gripper_width": matched_action[6:7],
    }
    gt_state = get_ground_truth_state(dataset, matched_idx)
    return matched_idx, used_horizon_index, pred_state, gt_state


def append_rollout_record(
    records,
    episode_id,
    current_idx,
    matched_idx,
    used_horizon_index,
    raw_action_horizon,
    abs_action_horizon,
    pred_state,
    gt_state,
):
    records.append({
        "episode_id": int(episode_id),
        "input_timestep": int(current_idx),
        "matched_timestep": int(matched_idx),
        "used_horizon_index": int(used_horizon_index),
        "raw_action_horizon": np.asarray(raw_action_horizon, dtype=np.float32),
        "abs_action_horizon": np.asarray(abs_action_horizon, dtype=np.float32),
        "pred_eef_pos": np.asarray(pred_state["eef_pos"], dtype=np.float32),
        "pred_eef_rot_axis_angle": np.asarray(
            pred_state["eef_rot_axis_angle"],
            dtype=np.float32,
        ),
        "pred_gripper_width": np.asarray(
            pred_state["gripper_width"],
            dtype=np.float32,
        ),
        "gt_eef_pos": np.asarray(gt_state["eef_pos"], dtype=np.float32),
        "gt_eef_rot_axis_angle": np.asarray(
            gt_state["eef_rot_axis_angle"],
            dtype=np.float32,
        ),
        "gt_gripper_width": np.asarray(
            gt_state["gripper_width"],
            dtype=np.float32,
        ),
    })

def pack_rollout_records(records):
    if len(records) == 0:
        return {}

    scalar_keys = [
        "episode_id",
        "input_timestep",
        "matched_timestep",
        "used_horizon_index",
    ]
    array_keys = [
        "raw_action_horizon",
        "abs_action_horizon",
        "pred_eef_pos",
        "pred_eef_rot_axis_angle",
        "pred_gripper_width",
        "gt_eef_pos",
        "gt_eef_rot_axis_angle",
        "gt_gripper_width",
    ]

    packed = {
        key: np.asarray([record[key] for record in records], dtype=np.int64)
        for key in scalar_keys
    }
    packed.update({
        key: np.stack([record[key] for record in records], axis=0)
        for key in array_keys
    })
    return packed


# ===== Composing Pipeline =====
def run_closed_loop_rollout(policy, dataset, cfg, device, episode_id, horizon_index):
    start_idx, end_idx, demo_start_pose = get_demo_info(dataset, episode_id)
    action_horizon = int(cfg.task.shape_meta.action.horizon)

    if horizon_index >= action_horizon:
        raise ValueError(
            f"horizon_index={horizon_index} is outside action horizon {action_horizon}."
        )

    current_idx = start_idx
    rollout_state = {}
    records = []

    policy.eval()
    policy.reset()
    while current_idx < end_idx - 1:
        print(
            f"rollout episode {episode_id}: "
            f"{current_idx - start_idx} / {end_idx - start_idx}",
            flush=True,
        )

        env_obs, model_obs = build_rollout_inputs(
            dataset,
            current_idx,
            start_idx,
            rollout_state,
            cfg,
            demo_start_pose,
        )
        raw_action_horizon, abs_action_horizon = predict_absolute_horizon(
            policy,
            model_obs,
            env_obs,
            cfg,
            device,
        )

        matched_idx, used_horizon_index, pred_state, gt_state = advance_rollout(
            dataset,
            rollout_state,
            current_idx,
            end_idx,
            abs_action_horizon,
            cfg,
            horizon_index,
        )

        append_rollout_record(
            records,
            episode_id,
            current_idx,
            matched_idx,
            used_horizon_index,
            raw_action_horizon,
            abs_action_horizon,
            pred_state,
            gt_state,
        )
        current_idx = matched_idx

    return pack_rollout_records(records)

def save_rollout_results(output_dir, rollout_results, cfg):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "closed_loop_predictions.npz",
        **rollout_results,
    )

    metadata = {
        "dataset_path": str(cfg.task.dataset_path),
        "action_horizon": int(cfg.task.shape_meta.action.horizon),
        "action_down_sample_steps": int(cfg.task.shape_meta.action.down_sample_steps),
        "num_rollout_steps": int(
            len(next(iter(rollout_results.values()))) if rollout_results else 0
        ),
        "result_keys": list(rollout_results.keys()),
    }
    with open(output_dir / "closed_loop_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ===== Main function =====
def main():
    print("ckpt:", CKPT_PATH)
    print("output:", OUTPUT_DIR)
    print("device:", DEVICE)
    print("episode_id:", EPISODE_ID)
    print("horizon_index:", HORIZON_INDEX)

    policy, cfg = load_policy(CKPT_PATH, DEVICE)
    dataset = load_dataset(cfg)
    print("dataset length:", len(dataset))

    rollout_results = run_closed_loop_rollout(
        policy,
        dataset,
        cfg,
        DEVICE,
        episode_id=EPISODE_ID,
        horizon_index=HORIZON_INDEX,
    )

    save_rollout_results(OUTPUT_DIR, rollout_results, cfg)
    print("saved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    os.chdir(str(ROOT_DIR))
    main()
