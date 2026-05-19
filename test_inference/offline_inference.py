'''
Offline inference on the grasp-cube UMI training dataset.
'''

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
from torch.utils.data import DataLoader

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
register_codecs()

# ===== Variables =====

CKPT_PATH = "data/outputs/2026.04.27/05.26.56_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt"
OUTPUT_DIR = "test_inference/results/grasp_cube"
DEVICE = "cuda:0"
BATCH_SIZE = 6
NUM_WORKERS = 0
MAX_BATCHES = None  # None means whole dataset; use 1 for quick test

# ===== Inference Functions =====

def load_policy(ckpt_path, device):
    # load checkpoint
    payload = torch.load(
        open(ckpt_path, "rb"),
        map_location="cpu",
        pickle_module=dill,
    )

    # cfg is the saved Hydra config from training
    cfg = payload["cfg"]

    # build workspace class from cfg._target_
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace

    # load model weights into workspace
    workspace.load_payload(
        payload,
        exclude_keys=None,
        include_keys=None,
    )

    # choose model
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # move to GPU and eval mode
    policy.to(torch.device(device))
    policy.eval()

    return policy, cfg

def create_dataloader(cfg, batch_size, num_workers):
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return dataset, dataloader

# Used some help with this one
def get_sample_metadata(dataset, dataset_indices):
    """
    Build analysis tags for the samples we just saved.

    Important distinction:
    - The original replay buffer stores robot/camera/action arrays.
    - The UmiDataset creates a SequenceSampler at runtime.
    - The SequenceSampler builds dataset.sampler.indices in memory.

    Each sampler index entry is:
        (current_idx, start_idx, end_idx, before_first_grasp)

    current_idx is the global replay-buffer timestep for this sample.
    start_idx and end_idx are the global timestep bounds of that demo.
    before_first_grasp is computed by the sampler from gripper width.
    It is not a manually labeled field in the original dataset.
    """
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:])

    metadata = {
        "dataset_index": [],
        "episode_id": [],
        "episode_timestep": [],
        "before_first_grasp": [],
    }

    for dataset_idx in dataset_indices:
        # Map a dataset sample number back to the replay-buffer window it uses.
        current_idx, start_idx, end_idx, before_first_grasp = (
            dataset.sampler.indices[int(dataset_idx)]
        )

        # episode_ends stores exclusive global end indices for each demo.
        # Example: [100, 250] means demo 0 is [0, 100),
        # and demo 1 is [100, 250).
        episode_id = np.searchsorted(episode_ends, current_idx, side="right")

        metadata["dataset_index"].append(int(dataset_idx))
        metadata["episode_id"].append(int(episode_id))

        # Convert global replay-buffer time into local time inside the demo.
        # Example: current_idx=120 and start_idx=100 -> demo timestep 20.
        metadata["episode_timestep"].append(int(current_idx - start_idx))
        metadata["before_first_grasp"].append(bool(before_first_grasp))

    return {
        "dataset_index": np.asarray(metadata["dataset_index"], dtype=np.int64),
        "episode_id": np.asarray(metadata["episode_id"], dtype=np.int64),
        "episode_timestep": np.asarray(metadata["episode_timestep"], dtype=np.int64),
        "before_first_grasp": np.asarray(metadata["before_first_grasp"], dtype=bool),
    }

def run_inference(policy, dataset, dataloader, device, max_batches=None):
    device = torch.device(device)
    policy.eval()
    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    all_pred = []
    all_gt = []
    all_metadata = {
        "dataset_index": [],
        "episode_id": [],
        "episode_timestep": [],
        "before_first_grasp": [],
    }

    # Because the dataloader uses shuffle=False, batches arrive in dataset order.
    # dataset_cursor tracks which dataset sample numbers are in the current batch.
    dataset_cursor = 0

    for batch_idx, batch in enumerate(dataloader):
        print(f"batch {batch_idx + 1} / {total_batches}", flush=True)

        batch = dict_apply(
            batch,
            lambda x: x.to(device, non_blocking=True),
        )

        with torch.no_grad():
            result = policy.predict_action(batch["obs"], None)

        pred_action = result["action_pred"]
        gt_action = batch["action"]

        all_pred.append(pred_action.detach().to("cpu").numpy())
        all_gt.append(gt_action.detach().to("cpu").numpy())

        this_batch_size = gt_action.shape[0]
        dataset_indices = np.arange(
            dataset_cursor,
            dataset_cursor + this_batch_size,
            dtype=np.int64,
        )
        batch_metadata = get_sample_metadata(dataset, dataset_indices)
        for key, value in batch_metadata.items():
            all_metadata[key].append(value)
        dataset_cursor += this_batch_size

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    pred_np = np.concatenate(all_pred, axis=0)
    gt_np = np.concatenate(all_gt, axis=0)
    metadata_np = {
        key: np.concatenate(value, axis=0)
        for key, value in all_metadata.items()
    }

    return pred_np, gt_np, metadata_np

def save_results(output_dir, pred, gt, sample_metadata, cfg):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "offline_predictions.npz",
        pred_action=pred,
        gt_action=gt,
        **sample_metadata,
    )

    metadata = {
        "dataset_path": str(cfg.task.dataset_path),
        "pred_shape": list(pred.shape),
        "gt_shape": list(gt.shape),
        "action_horizon": int(cfg.task.shape_meta.action.horizon),
        "sample_metadata_keys": list(sample_metadata.keys()),
        "num_unique_episodes": int(np.unique(sample_metadata["episode_id"]).size),
        "episode_ids": [
            int(x) for x in np.unique(sample_metadata["episode_id"])
        ],
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

# No need for plots or error calculations at this moment, let's just inspect the raw ground truth and predictions to see if reasonable first
# Maybe compare with 50 epochs checkpoint
# The error calculations and plots will be in a seperate jupyter notebook for easier iteration

# ===== Main function =====

def main():
    print("ckpt:", CKPT_PATH)
    print("output:", OUTPUT_DIR)
    print("batch_size:", BATCH_SIZE)
    print("num_workers:", NUM_WORKERS)
    print("device:", DEVICE)

    policy, cfg = load_policy(CKPT_PATH, DEVICE)

    dataset, dataloader = create_dataloader(
        cfg,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print("dataset length:", len(dataset))

    pred, gt, sample_metadata = run_inference(
        policy,
        dataset,
        dataloader,
        DEVICE,
        max_batches=MAX_BATCHES,
    )

    print("pred shape:", pred.shape)
    print("gt shape:", gt.shape)
    print("episode ids:", np.unique(sample_metadata["episode_id"]))

    save_results(OUTPUT_DIR, pred, gt, sample_metadata, cfg)
    print("saved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    os.chdir(str(ROOT_DIR))
    main()
