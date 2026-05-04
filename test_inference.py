import argparse
import json
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


DEFAULT_CKPT = (
    "data/outputs/2026.04.27/05.26.56_train_diffusion_unet_timm_umi/"
    "checkpoints/latest.ckpt"
)


class IndexedDataset(Dataset):
    """Wrap a dataset so each collated batch keeps the original dataset index."""

    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx = int(self.indices[idx])
        item = self.dataset[dataset_idx]
        item["sample_index"] = torch.tensor(dataset_idx, dtype=torch.long)
        return item


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained UMI checkpoint on the same training dataset specified "
            "inside the checkpoint config and compare predicted actions to the "
            "dataset actions."
        )
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to a .ckpt file.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for inference, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=32,
        help="Number of training samples to evaluate when --episode-id is not set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of independent observations to process at once. "
            "Default 1 matches real-robot deployment."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=["random", "sequential"],
        default="random",
        help="How to choose samples from the training dataset.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index used when --sampling sequential.",
    )
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated dataset indices. Overrides --sampling and --sample-count.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help=(
            "Evaluate every valid training-dataset timestep from this replay-buffer "
            "episode. Overrides --sample-indices, --sampling, and --sample-count."
        ),
    )
    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="List available training episodes and valid timestep counts, then exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for dataset sampling, image augmentation, and diffusion noise.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 for deterministic debugging.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override policy.num_inference_steps. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--print-steps",
        type=int,
        default=5,
        help="Number of action horizon steps to print for compact terminal preview.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=3,
        help="Number of evaluated timesteps to preview in the terminal.",
    )
    parser.add_argument(
        "--save-npz",
        default=None,
        help="Optional path for saving pred_action, gt_action, and sample indices.",
    )
    parser.add_argument(
        "--save-summary",
        default=None,
        help="Optional path for saving the metric summary as JSON.",
    )
    parser.add_argument(
        "--save-jsonl",
        default=None,
        help=(
            "Optional path for compact per-timestep JSONL records containing "
            "pred_action, gt_action, error_pred_minus_gt, and action_mse."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise ValueError(
            f"test_inference.py is GPU-only for this diagnostic; got --device {device_arg!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this diagnostic, but torch.cuda.is_available() is false."
        )
    return device


def load_workspace_and_policy(ckpt_path: Path, device: torch.device):
    # The training entrypoint registers this resolver before Hydra config use.
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    payload = torch.load(ckpt_path.open("rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return workspace, policy, cfg


def load_training_dataset(cfg):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


def episode_index_map(dataset):
    metadata = sample_metadata(dataset, np.arange(len(dataset), dtype=np.int64))
    episode_map = {}
    for item in metadata:
        episode_id = item["episode_id"]
        if episode_id is None:
            continue
        episode_map.setdefault(episode_id, []).append(item["dataset_index"])
    return episode_map


def print_available_episodes(dataset):
    episode_map = episode_index_map(dataset)
    print("\nAvailable training episodes")
    print("---------------------------")
    for episode_id in sorted(episode_map):
        indices = episode_map[episode_id]
        meta = sample_metadata(dataset, [indices[0]])[0]
        print(
            f"episode_id={episode_id:03d} "
            f"valid_timesteps={len(indices):04d} "
            f"dataset_index_range={indices[0]}..{indices[-1]} "
            f"replay_index_range={meta['episode_start_index']}..{meta['episode_end_index'] - 1}"
        )


def choose_indices(dataset, args):
    dataset_len = len(dataset)
    if args.episode_id is not None:
        episode_map = episode_index_map(dataset)
        if args.episode_id not in episode_map:
            available = ", ".join(str(x) for x in sorted(episode_map))
            raise ValueError(
                f"Episode {args.episode_id} has no valid training timesteps. "
                f"Available training episodes: {available}"
            )
        return episode_map[args.episode_id]

    if args.sample_indices is not None:
        indices = [
            int(idx.strip())
            for idx in args.sample_indices.split(",")
            if idx.strip()
        ]
    elif args.sampling == "sequential":
        end = min(dataset_len, args.start_index + args.sample_count)
        indices = list(range(args.start_index, end))
    else:
        rng = np.random.default_rng(args.seed)
        count = min(args.sample_count, dataset_len)
        indices = rng.choice(dataset_len, size=count, replace=False).tolist()

    bad = [idx for idx in indices if idx < 0 or idx >= dataset_len]
    if bad:
        raise ValueError(f"Dataset indices out of range 0..{dataset_len - 1}: {bad}")
    return indices


def summarize_actions(pred: np.ndarray, gt: np.ndarray):
    diff = pred - gt
    summary = {
        "action_mse": float(np.mean(diff**2)),
        "action_mae": float(np.mean(np.abs(diff))),
        "action_max_abs_error": float(np.max(np.abs(diff))),
    }
    return summary


def sample_metadata(dataset, sample_indices):
    sampler = getattr(dataset, "sampler", None)
    replay_buffer = getattr(dataset, "replay_buffer", None)
    if sampler is None or not hasattr(sampler, "indices"):
        return []

    episode_ends = None
    if replay_buffer is not None and hasattr(replay_buffer, "episode_ends"):
        episode_ends = np.asarray(replay_buffer.episode_ends[:])

    metadata = []
    for dataset_idx in sample_indices:
        current_idx, start_idx, end_idx, before_first_grasp = sampler.indices[dataset_idx]
        episode_id = None
        if episode_ends is not None:
            matches = np.flatnonzero(episode_ends == end_idx)
            if len(matches) > 0:
                episode_id = int(matches[0])
        metadata.append(
            {
                "dataset_index": int(dataset_idx),
                "episode_id": episode_id,
                "current_replay_index": int(current_idx),
                "episode_start_index": int(start_idx),
                "episode_end_index": int(end_idx),
                "before_first_grasp": bool(before_first_grasp),
            }
        )
    return metadata


def compact_record(pred_i, gt_i, sample_index, metadata):
    metadata = metadata or {}
    diff_i = pred_i - gt_i
    return {
        "dataset_index": int(sample_index),
        "episode_id": metadata.get("episode_id"),
        "current_replay_index": metadata.get("current_replay_index"),
        "pred_action": pred_i.tolist(),
        "gt_action": gt_i.tolist(),
        "error_pred_minus_gt": diff_i.tolist(),
        "action_mse": float(np.mean(diff_i**2)),
    }


def print_summary(summary):
    print("\nAction match metrics on training samples")
    print("----------------------------------------")
    for key, value in summary.items():
        print(f"{key:42s} {value:.10f}")


def build_compact_records(pred, gt, sample_indices, metadata):
    return [
        compact_record(pred_i, gt_i, sample_index, meta)
        for pred_i, gt_i, sample_index, meta in zip(pred, gt, sample_indices, metadata)
    ]


def print_compact_preview(records, print_samples, print_steps):
    np.set_printoptions(suppress=True, precision=5)
    print("\nCompact per-timestep preview")
    print("----------------------------")
    print("action columns: rel_pos[0:3], rel_rot6d[3:9], gripper_width[9]")
    for record in records[:print_samples]:
        pred = np.asarray(record["pred_action"], dtype=np.float32)
        gt = np.asarray(record["gt_action"], dtype=np.float32)
        err = np.asarray(record["error_pred_minus_gt"], dtype=np.float32)
        n_steps = min(print_steps, pred.shape[0])
        print(
            f"\ndataset_index={record['dataset_index']} "
            f"episode_id={record['episode_id']} "
            f"current_replay_index={record['current_replay_index']} "
            f"action_mse={record['action_mse']:.10f}"
        )
        print(f"pred_action first {n_steps} steps:\n{pred[:n_steps]}")
        print(f"gt_action first {n_steps} steps:\n{gt[:n_steps]}")
        print(f"error_pred_minus_gt first {n_steps} steps:\n{err[:n_steps]}")


def write_jsonl(path, records):
    jsonl_path = Path(path).expanduser()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Saved per-timestep JSONL to: {jsonl_path}")


def main():
    args = parse_args()
    seed_everything(args.seed)

    ckpt_path = Path(args.ckpt).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    device = resolve_device(args.device)
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Using device: {device}")
    _, policy, cfg = load_workspace_and_policy(ckpt_path, device)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps

    print(f"Checkpoint dataset_path: {cfg.task.dataset.dataset_path}")
    print(f"Policy inference steps: {policy.num_inference_steps}")
    print(f"EMA policy: {bool(cfg.training.use_ema)}")

    dataset = load_training_dataset(cfg)
    if args.list_episodes:
        print_available_episodes(dataset)
        return

    sample_indices = choose_indices(dataset, args)
    indexed_dataset = IndexedDataset(dataset, sample_indices)
    dataloader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Training dataset samples available: {len(dataset)}")
    print(f"Evaluating samples: {len(sample_indices)}")

    pred_batches = []
    gt_batches = []
    seen_indices = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            obs = dict_apply(batch["obs"], lambda x: x.to(device, non_blocking=True))
            gt_action = batch["action"].to(device, non_blocking=True)

            # Reset the RNG before each batch so diffusion sampling and the
            # checkpoint's stochastic image transforms are reproducible.
            torch.manual_seed(args.seed + batch_idx)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + batch_idx)
            policy.reset()
            pred_action = policy.predict_action(obs, None)["action_pred"]

            pred_batches.append(pred_action.detach().cpu().numpy())
            gt_batches.append(gt_action.detach().cpu().numpy())
            seen_indices.extend(batch["sample_index"].cpu().numpy().tolist())

    pred = np.concatenate(pred_batches, axis=0)
    gt = np.concatenate(gt_batches, axis=0)
    seen_indices = np.asarray(seen_indices, dtype=np.int64)
    metadata = sample_metadata(dataset, seen_indices)
    records = build_compact_records(pred, gt, seen_indices, metadata)
    summary = summarize_actions(pred, gt)
    summary.update(
        {
            "num_samples": int(pred.shape[0]),
            "action_horizon": int(pred.shape[1]),
            "action_dim": int(pred.shape[2]),
        }
    )

    print_summary(summary)
    print_compact_preview(records, args.print_samples, args.print_steps)

    if args.save_npz is not None:
        npz_path = Path(args.save_npz).expanduser()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "pred_action": pred,
            "gt_action": gt,
            "action_error_pred_minus_gt": pred - gt,
            "sample_indices": seen_indices,
            "metadata": np.asarray(metadata, dtype=object),
        }
        np.savez_compressed(
            npz_path,
            **save_data,
        )
        print(f"\nSaved arrays to: {npz_path}")

    if args.save_summary is not None:
        summary_path = Path(args.save_summary).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(
                {
                    "summary": summary,
                    "metadata": metadata,
                    "preview_records": records[:args.print_samples],
                },
                f,
                indent=2,
            )
        print(f"Saved summary to: {summary_path}")

    if args.save_jsonl is not None:
        write_jsonl(args.save_jsonl, records)


if __name__ == "__main__":
    main()
