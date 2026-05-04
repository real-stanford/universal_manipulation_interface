import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
from dataclasses import dataclass
from typing import Optional

import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import matplotlib

if ('DISPLAY' not in os.environ) and ('WAYLAND_DISPLAY' not in os.environ):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


@dataclass
class Trajectory:
    name: str
    poses: np.ndarray
    total_frames: int
    lost_frames: int
    csv_path: pathlib.Path


def load_tx_tag_slam(session_dir: pathlib.Path) -> Optional[np.ndarray]:
    tx_path = session_dir.joinpath('demos', 'mapping', 'tx_slam_tag.json')
    if not tx_path.is_file():
        return None
    tx_slam_tag = np.array(json.load(tx_path.open('r'))['tx_slam_tag'])
    return np.linalg.inv(tx_slam_tag)


def pose_array_from_df(df: pd.DataFrame) -> np.ndarray:
    pos = df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    quat = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy(dtype=np.float64)
    rot = Rotation.from_quat(quat).as_matrix()

    poses = np.zeros((len(df), 4, 4), dtype=np.float64)
    poses[:, 3, 3] = 1.0
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = pos
    return poses


def load_trajectory(csv_path: pathlib.Path, tx_base_slam: Optional[np.ndarray]) -> Optional[Trajectory]:
    df = pd.read_csv(csv_path)
    total_frames = len(df)
    if total_frames == 0:
        return None

    is_lost = df['is_lost'].astype(str).str.lower().isin(['1', 'true', 't', 'yes'])
    valid_df = df.loc[~is_lost].copy()
    lost_frames = total_frames - len(valid_df)
    if len(valid_df) == 0:
        return None

    poses = pose_array_from_df(valid_df)
    if tx_base_slam is not None:
        poses = tx_base_slam @ poses

    return Trajectory(
        name=csv_path.parent.name,
        poses=poses,
        total_frames=total_frames,
        lost_frames=lost_frames,
        csv_path=csv_path
    )


def get_mapping_csv(mapping_dir: pathlib.Path) -> Optional[pathlib.Path]:
    for name in ('camera_trajectory.csv', 'mapping_camera_trajectory.csv'):
        path = mapping_dir.joinpath(name)
        if path.is_file():
            return path
    return None


def set_axes_equal(ax: plt.Axes, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius <= 0:
        radius = 0.1

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


@click.command()
@click.argument('session_dir', type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option('--frame', type=click.Choice(['auto', 'slam', 'tag']), default='auto', show_default=True,
              help='Coordinate frame to visualize.')
@click.option('--save', type=click.Path(dir_okay=False, path_type=pathlib.Path), default=None,
              help='Optional output image path.')
@click.option('--max_demos', type=int, default=None, help='Limit number of demo trajectories plotted.')
@click.option('--stride', type=int, default=1, show_default=True, help='Subsample plotted points.')
@click.option('--hide_mapping', is_flag=True, default=False, help='Do not draw mapping trajectory.')
@click.option('--elev', type=float, default=24.0, show_default=True, help='3D view elevation in degrees.')
@click.option('--azim', type=float, default=-60.0, show_default=True, help='3D view azimuth in degrees.')
def main(session_dir: pathlib.Path, frame: str, save: Optional[pathlib.Path], max_demos: Optional[int],
         stride: int, hide_mapping: bool, elev: float, azim: float) -> None:
    session_dir = session_dir.absolute()
    demos_dir = session_dir.joinpath('demos')
    mapping_dir = demos_dir.joinpath('mapping')
    mapping_csv = get_mapping_csv(mapping_dir)

    tx_tag_slam = load_tx_tag_slam(session_dir)
    if frame == 'tag':
        if tx_tag_slam is None:
            raise click.ClickException('Requested --frame tag but tx_slam_tag.json was not found.')
        tx_base_slam = tx_tag_slam
        frame_name = 'tag'
    elif frame == 'slam':
        tx_base_slam = None
        frame_name = 'slam'
    else:
        tx_base_slam = tx_tag_slam
        frame_name = 'tag' if tx_tag_slam is not None else 'slam'

    demo_csvs = sorted(demos_dir.glob('demo_*/camera_trajectory.csv'))
    if max_demos is not None:
        demo_csvs = demo_csvs[:max_demos]

    demo_trajs = []
    skipped = []
    for csv_path in demo_csvs:
        traj = load_trajectory(csv_path, tx_base_slam=tx_base_slam)
        if traj is None:
            skipped.append(csv_path.parent.name)
            continue
        demo_trajs.append(traj)

    mapping_traj = None
    if mapping_csv is not None and not hide_mapping:
        mapping_traj = load_trajectory(mapping_csv, tx_base_slam=tx_base_slam)

    if (mapping_traj is None) and (not demo_trajs):
        raise click.ClickException('No valid trajectories found to plot.')

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    all_xyz = []

    if mapping_traj is not None:
        xyz = mapping_traj.poses[::stride, :3, 3]
        all_xyz.append(xyz)
        ax.plot(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            color='black', linewidth=2.2, alpha=0.85, label='mapping'
        )
        ax.scatter(
            xyz[0, 0], xyz[0, 1], xyz[0, 2],
            color='black', marker='o', s=28
        )

    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(demo_trajs), 1))
    for idx, traj in enumerate(demo_trajs):
        xyz = traj.poses[::stride, :3, 3]
        all_xyz.append(xyz)
        color = cmap(idx % cmap.N)
        label = f'{traj.name} ({traj.lost_frames}/{traj.total_frames} lost)'
        ax.plot(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            color=color, linewidth=1.3, alpha=0.9, label=label
        )
        ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color=color, marker='o', s=12)
        ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=color, marker='x', s=16)

    xyz_all = np.concatenate(all_xyz, axis=0)
    set_axes_equal(ax, xyz_all)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'{session_dir.name} trajectories in {frame_name} frame')

    if len(demo_trajs) <= 12:
        ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()

    print(f'Session: {session_dir}')
    print(f'Frame: {frame_name}')
    print(f'Demos plotted: {len(demo_trajs)}')
    if skipped:
        print(f'Skipped demos with no valid tracked frames: {len(skipped)}')
        for name in skipped:
            print(f'  {name}')
    if mapping_csv is not None:
        print(f'Mapping CSV: {mapping_csv}')

    if save is not None:
        save = save.absolute()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=180)
        print(f'Saved figure to {save}')
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
