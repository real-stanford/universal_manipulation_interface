import collections
import copy
import math
import os
import pathlib
from typing import Dict, List, Optional

import dill
import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import tqdm
import wandb.sdk.data_types.video as wv

import wandb
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.pose_repr_util import compute_relative_pose
from diffusion_policy.env.robomimic.robomimic_image_wrapper import \
    RobomimicImageWrapper
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder, VideoRecordingWrapper)
from diffusion_policy.model.common.rotation_transformer import \
    RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        if key.endswith('pos_abs'):
            continue
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    
    for key, attr in shape_meta['render'].items():
        assert attr['type'] == 'rgb'
        if not key in modality_mapping['rgb']:
            modality_mapping['rgb'].append(key)

    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_action_steps=8,
            pose_repr: dict={},
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            tqdm_interval_sec=5.0,
            n_envs=None,
            obs_latency_steps=0
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        # use posiiton control
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        
        # horizion & rotation transformers
        self.rot_quat2mat = RotationTransformer(
            from_rep='quaternion',
            to_rep='matrix')
        self.rot_aa2mat = RotationTransformer(
            from_rep='axis_angle',
            to_rep='matrix')
        self.rot_mat2target = dict()
        self.key_horizon = dict()
        for key, attr in shape_meta['obs'].items():
            self.key_horizon[key] = shape_meta['obs'][key]['horizon']
            if 'rotation_rep' in attr:
                self.rot_mat2target[key] = RotationTransformer(
                    from_rep='matrix',
                    to_rep=attr['rotation_rep'])
        max_obs_horizon = max(self.key_horizon.values())
        
        self.rot_quat2euler = RotationTransformer(
            from_rep='quaternion',
            to_rep='euler_angles', to_convention='XYZ')

        assert 'rotation_rep' in shape_meta['action']
        self.rot_mat2target['action'] = RotationTransformer(
            from_rep='matrix',
            to_rep=shape_meta['action']['rotation_rep'])
        self.key_horizon['action'] = shape_meta['action']['horizon']

        self.shape_meta = shape_meta
        self.obs_latency_steps = obs_latency_steps

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=max_obs_horizon + obs_latency_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=max_obs_horizon + obs_latency_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_excuted_actions = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            
            # without latency
            # |t-1            |t
            # |a|a|a|a|a|a|a|a|p|p|p|p|p|p|p|p|
            #                 |a|a|a|a|a|a|a|a|p|p|p|p|p|p|p|p|
            #
            # with latency = 2
            #     |t-1            |t
            # |x|x|a|a|a|a|a|a|a|a|p|p|p|p|p|p|
            #                 |x|x|a|a|a|a|a|a|p|p|p|p|p|p|p|p|  

            prev_action = None # the last latency_steps actions
            done = False
            this_excuted_actions = [list() for _ in range(end-start)]
            while not done:
                # create obs dict
                obs_dict = dict()
                for key in obs.keys():
                    slice_start = -(self.key_horizon[key] + self.obs_latency_steps)
                    slice_end = None if self.obs_latency_steps == 0 else -self.obs_latency_steps
                    obs_dict[key] = obs[key][:, slice_start: slice_end]

                # add additional obs
                key = 'robot0_eef_pos_abs'
                if key in self.shape_meta['obs']:
                    axis = self.shape_meta['obs'][key]['axis']
                    if isinstance(axis, int):
                        axis = [axis]
                    obs_dict[key] = obs[key[:-4]][:, -self.key_horizon[key]:, axis]
                
                key = 'robot0_eef_quat_abs'
                if key in self.shape_meta['obs']:
                    axis = self.shape_meta['obs'][key]['axis']
                    if isinstance(axis, int):
                        axis = [axis]
                    obs_dict[key] = self.rot_quat2euler.forward(obs[key[:-4]])[:, -self.key_horizon[key]:, list(axis)]

                # solve rel action
                current_pos = copy.copy(obs_dict['robot0_eef_pos'][:, -1:])
                current_rot_mat = copy.copy(self.rot_quat2mat.forward(obs_dict['robot0_eef_quat'][:, -1:]))

                # solve relative obs
                obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'] = compute_relative_pose(
                    pos=obs_dict['robot0_eef_pos'],
                    rot=obs_dict['robot0_eef_quat'],
                    base_pos=current_pos if self.obs_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
                    base_rot_mat=current_rot_mat if self.obs_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
                    rot_transformer_to_mat=self.rot_quat2mat,
                    rot_transformer_to_target=self.rot_mat2target['robot0_eef_quat']
                )

                # device transfer
                obs_dict = dict_apply(obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                fixed_action_prefix = None
                if prev_action is not None:
                    action_pos, action_rot = compute_relative_pose(
                        pos=prev_action[..., :3],
                        rot=prev_action[..., 3: -1],
                        base_pos=current_pos if self.action_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
                        base_rot_mat=current_rot_mat if self.action_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
                        rot_transformer_to_mat=self.rot_aa2mat,
                        rot_transformer_to_target=self.rot_mat2target['action']
                    )
                    action_gripper = prev_action[..., -1:]
                    fixed_action_prefix = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)
                    fixed_action_prefix = torch.from_numpy(fixed_action_prefix).to(device=device)

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, fixed_action_prefix)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                
                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # action rotation transformer
                action_pos, action_rot = compute_relative_pose(
                    pos=action[..., :3],
                    rot=action[..., 3: -1],
                    base_pos=current_pos if self.action_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
                    base_rot_mat=current_rot_mat if self.action_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
                    rot_transformer_to_mat=self.rot_aa2mat,
                    rot_transformer_to_target=self.rot_mat2target['action'],
                    backward=True
                )
                action_gripper = action[..., -1:]
                action_all = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)

                env_action = action_all[:, self.obs_latency_steps: self.obs_latency_steps + self.n_action_steps, :]
                prev_action = env_action[:, -self.obs_latency_steps:, :]

                for idx, this_done in enumerate(env.call('get_attr', 'done')):
                    if not np.any(this_done):
                        this_excuted_actions[idx].append(env_action[idx])
                        # this_excuted_actions[idx].append(action_all[idx])

                # step env
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)


                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_excuted_actions[this_global_slice] = [np.concatenate(x) for x in this_excuted_actions]

        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix+f'sim_executed_action_{seed}'] = all_excuted_actions[i]

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data