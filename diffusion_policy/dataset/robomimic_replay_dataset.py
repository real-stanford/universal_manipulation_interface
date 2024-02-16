import concurrent.futures
import copy
import multiprocessing
import os
import shutil
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits
from tqdm import tqdm, trange

from diffusion_policy.codecs.imagecodecs_numcodecs import (Jpeg2k,
                                                           register_codecs)
from diffusion_policy.common.normalize_util import (
    array_to_stats, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat,
    concatenate_normalizer)
from diffusion_policy.common.pose_repr_util import compute_relative_pose
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.model.common.rotation_transformer import \
    RotationTransformer

register_codecs()


class RobomimicReplayDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        pose_repr: dict={},
        temporally_independent_normalization: bool=False,
        seed: int=42,
        val_ratio: float=0.0,
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')

        replay_buffer = _convert_robomimic_to_replay(
            store=zarr.MemoryStore(), 
            shape_meta=shape_meta, 
            dataset_path=dataset_path)
        
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        rot_mat2target = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            # solve obs_horizon
            horizon = attr['horizon']
            key_horizon[key] = horizon

            # solve latency and down_sample
            assert not 'latency_steps' in attr
            assert not 'down_sample_steps' in attr
            key_latency_steps[key] = 0
            key_down_sample_steps[key] = 1
        
            # solve rotation transformer
            if 'rotation_rep' in attr:
                rot_mat2target[key] = RotationTransformer(
                    from_rep='matrix', to_rep=attr['rotation_rep'])

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = 0
        key_down_sample_steps['action'] = 1
        rot_mat2target['action'] = RotationTransformer(
            from_rep='matrix',
            to_rep=shape_meta['action']['rotation_rep'])

        self.rot_quat2mat = RotationTransformer(
            from_rep='quaternion', to_rep='matrix')
        self.rot_aa2mat = RotationTransformer(
            from_rep='axis_angle', to_rep='matrix')


        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask
        )
        
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.rot_mat2target = rot_mat2target
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False


    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        for idx in trange(len(self.sampler), desc='iterating dataset to get normalization'):
            this_data = self[idx]
            for key in self.lowdim_keys:
                data_cache[key].append(this_data['obs'][key])
            data_cache['action'].append(this_data['action'])
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.stack(data_cache[key])
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        assert data_cache['action'].shape[-1] <= 10 # only handle single arm now
        this_normalizer = concatenate_normalizer([
            get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., :3])),      # pos
            get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., 3:-1])), # rot
            get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., -1:]))      # gripper
        ])
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'unsupported {key}' )
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        # get current pose
        current_pos = copy.copy(obs_dict['robot0_eef_pos'][-1])
        current_rot_mat = copy.copy(self.rot_quat2mat.forward(obs_dict['robot0_eef_quat'][-1]))
        
        # solve relative obs
        obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'] = compute_relative_pose(
            pos=obs_dict['robot0_eef_pos'],
            rot=obs_dict['robot0_eef_quat'],
            base_pos=current_pos if self.obs_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
            base_rot_mat=current_rot_mat if self.obs_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
            rot_transformer_to_mat=self.rot_quat2mat,
            rot_transformer_to_target=self.rot_mat2target['robot0_eef_quat']
        )
        
        # solve relative action
        action_pos, action_rot = compute_relative_pose(
            pos=data['action'][..., :3],
            rot=data['action'][..., 3:-1],
            base_pos=current_pos if self.action_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
            base_rot_mat=current_rot_mat if self.action_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
            rot_transformer_to_mat=self.rot_aa2mat,
            rot_transformer_to_target=self.rot_mat2target['action']
        )
        action_gripper = data['action'][..., -1:]
        data['action'] = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            if key.endswith('_abs'):
                data_key = data_key[:-4] # remove "_abs"
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            
            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                this_data = this_data[:, list(axis)]
            
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer