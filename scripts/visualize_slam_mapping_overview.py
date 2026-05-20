import json
import os
import pathlib
import pickle
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation

if ("DISPLAY" not in os.environ) and ("WAYLAND_DISPLAY" not in os.environ):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from umi.common.pose_util import pose_to_mat


@dataclass
class Trajectory:
    name: str
    xyz: np.ndarray
    total_frames: int
    lost_frames: int
    is_mapping: bool = False


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["1", "true", "t", "yes"])


def load_tx_tag_slam(session_dir: pathlib.Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    tx_path = session_dir.joinpath("demos", "mapping", "tx_slam_tag.json")
    if not tx_path.is_file():
        return None, None
    tx_slam_tag = np.array(json.load(tx_path.open("r"))["tx_slam_tag"], dtype=np.float64)
    tx_tag_slam = np.linalg.inv(tx_slam_tag)
    return tx_slam_tag, tx_tag_slam


def load_csv_xyz(
    csv_path: pathlib.Path,
    tx_tag_slam: Optional[np.ndarray],
    stride: int,
    is_mapping: bool = False,
) -> Optional[Trajectory]:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    is_lost = bool_series(df["is_lost"])
    valid_df = df.loc[~is_lost]
    if len(valid_df) == 0:
        return None

    pos = valid_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    quat = valid_df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=np.float64)
    rot = Rotation.from_quat(quat).as_matrix()

    poses = np.zeros((len(valid_df), 4, 4), dtype=np.float64)
    poses[:, 3, 3] = 1.0
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = pos
    if tx_tag_slam is not None:
        poses = tx_tag_slam @ poses

    return Trajectory(
        name=csv_path.parent.name,
        xyz=poses[::stride, :3, 3],
        total_frames=len(df),
        lost_frames=int(is_lost.sum()),
        is_mapping=is_mapping,
    )


def load_mapping_trajectory(
    session_dir: pathlib.Path,
    tx_tag_slam: Optional[np.ndarray],
    stride: int,
) -> Optional[Trajectory]:
    mapping_dir = session_dir.joinpath("demos", "mapping")
    for name in ["camera_trajectory.csv", "mapping_camera_trajectory.csv"]:
        csv_path = mapping_dir.joinpath(name)
        if csv_path.is_file():
            return load_csv_xyz(csv_path, tx_tag_slam, stride=stride, is_mapping=True)
    return None


def load_demo_trajectories(
    session_dir: pathlib.Path,
    tx_tag_slam: Optional[np.ndarray],
    stride: int,
    max_demos: Optional[int],
) -> list[Trajectory]:
    demo_csvs = sorted(session_dir.joinpath("demos").glob("demo_*/camera_trajectory.csv"))
    if max_demos is not None:
        demo_csvs = demo_csvs[:max_demos]

    out = []
    for csv_path in demo_csvs:
        traj = load_csv_xyz(csv_path, tx_tag_slam, stride=stride)
        if traj is not None:
            out.append(traj)
    return out


def load_tcp_trajectories(
    session_dir: pathlib.Path,
    stride: int,
    max_episodes: Optional[int],
) -> list[Trajectory]:
    plan_path = session_dir.joinpath("dataset_plan.pkl")
    if not plan_path.is_file():
        return []

    plan = pickle.load(plan_path.open("rb"))
    if max_episodes is not None:
        plan = plan[:max_episodes]

    out = []
    for ep_idx, episode in enumerate(plan):
        for gripper_idx, gripper in enumerate(episode["grippers"]):
            poses = pose_to_mat(gripper["tcp_pose"])
            out.append(
                Trajectory(
                    name=f"episode {ep_idx:03d} tcp{gripper_idx}",
                    xyz=poses[::stride, :3, 3],
                    total_frames=len(poses),
                    lost_frames=0,
                )
            )
    return out


def set_equal_2d(ax: plt.Axes, points: np.ndarray, dims: tuple[int, int], pad: float = 0.04) -> None:
    xy = points[:, dims]
    mins = np.nanmin(xy, axis=0)
    maxs = np.nanmax(xy, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    radius *= 1.0 + pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def set_equal_3d(ax: plt.Axes, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def concat_xyz(groups: Iterable[Trajectory]) -> np.ndarray:
    arrays = [traj.xyz for traj in groups if len(traj.xyz) > 0]
    if not arrays:
        return np.zeros((1, 3), dtype=np.float64)
    return np.concatenate(arrays, axis=0)


def add_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, fc: str) -> None:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#263238",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.8,
        color="#172026",
        linespacing=1.25,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#455a64",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arrow)


def draw_pipeline_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("SLAM pipeline files", loc="left", fontsize=13, fontweight="bold")
    boxes = [
        ((0.02, 0.70), "raw_videos/*.mp4\n00_process_videos.py", "#e8f1fb"),
        ((0.27, 0.70), "demos/*/raw_video.mp4\n01_extract_gopro_imu.py", "#e8f1fb"),
        ((0.52, 0.70), "mapping/map_atlas.osa\n02_create_map.py", "#eaf5e8"),
        ((0.77, 0.70), "camera_trajectory.csv\n03_batch_slam.py", "#eaf5e8"),
        ((0.27, 0.38), "tag_detection.pkl\n04_detect_aruco.py", "#fff4db"),
        ((0.52, 0.38), "tx_slam_tag.json\n05_run_calibrations.py", "#fff4db"),
        ((0.77, 0.38), "dataset_plan.pkl\n06_generate_dataset_plan.py", "#f1eafb"),
        ((0.77, 0.08), "dataset.zarr.zip\n07_generate_replay_buffer.py", "#fce8e6"),
    ]
    for xy, text, fc in boxes:
        add_box(ax, xy, 0.20, 0.17, text, fc)

    add_arrow(ax, (0.22, 0.785), (0.27, 0.785))
    add_arrow(ax, (0.47, 0.785), (0.52, 0.785))
    add_arrow(ax, (0.72, 0.785), (0.77, 0.785))
    add_arrow(ax, (0.37, 0.70), (0.37, 0.55))
    add_arrow(ax, (0.47, 0.47), (0.52, 0.47))
    add_arrow(ax, (0.72, 0.47), (0.77, 0.47))
    add_arrow(ax, (0.87, 0.38), (0.87, 0.25))


def draw_transform_panel(ax: plt.Axes, tx_slam_tag: Optional[np.ndarray]) -> None:
    ax.set_axis_off()
    ax.set_title("Coordinate mapping", loc="left", fontsize=13, fontweight="bold")
    add_box(ax, (0.03, 0.64), 0.21, 0.16, "SLAM map\nORB-SLAM3 atlas", "#eaf5e8")
    add_box(ax, (0.39, 0.64), 0.21, 0.16, "Camera pose\nT_slam_cam", "#e8f1fb")
    add_box(ax, (0.75, 0.64), 0.21, 0.16, "Aruco table tag\nT_cam_tag", "#fff4db")
    add_box(ax, (0.13, 0.30), 0.34, 0.17, "Table-tag frame\nT_tag_slam = inv(T_slam_tag)", "#fff4db")
    add_box(ax, (0.58, 0.30), 0.38, 0.17, "TCP trajectory\nT_tag_tcp = T_tag_slam @\nT_slam_cam @ T_cam_tcp", "#f1eafb")
    add_box(ax, (0.26, 0.04), 0.46, 0.16, "Policy dataset\nrobot*_eef_pos, robot*_eef_rot,\ngripper_width, camera*_rgb", "#fce8e6")

    add_arrow(ax, (0.24, 0.72), (0.39, 0.72))
    add_arrow(ax, (0.60, 0.72), (0.75, 0.72))
    add_arrow(ax, (0.49, 0.64), (0.36, 0.47))
    add_arrow(ax, (0.47, 0.385), (0.58, 0.385))
    add_arrow(ax, (0.77, 0.30), (0.52, 0.20))

    if tx_slam_tag is not None:
        t = tx_slam_tag[:3, 3]
        msg = f"T_slam_tag translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
    else:
        msg = "No tx_slam_tag.json found, plots stay in SLAM frame"
    ax.text(0.03, 0.91, msg, fontsize=8.8, color="#37474f", transform=ax.transAxes)


def plot_2d_view(
    ax: plt.Axes,
    trajectories: list[Trajectory],
    tcp_trajectories: list[Trajectory],
    dims: tuple[int, int],
    labels: tuple[str, str],
    title: str,
) -> None:
    cmap = matplotlib.colormaps["viridis"]
    demo_trajs = [t for t in trajectories if not t.is_mapping]
    mapping = next((t for t in trajectories if t.is_mapping), None)
    n = max(len(demo_trajs), 1)

    for idx, traj in enumerate(demo_trajs):
        color = cmap(0.15 + 0.72 * idx / max(n - 1, 1))
        xy = traj.xyz[:, dims]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.8, alpha=0.45)
        ax.scatter(xy[0, 0], xy[0, 1], color=color, s=6, alpha=0.55)

    if mapping is not None:
        xy = mapping.xyz[:, dims]
        ax.plot(xy[:, 0], xy[:, 1], color="#111111", linewidth=2.0, alpha=0.9, label="mapping video")

    for idx, traj in enumerate(tcp_trajectories):
        xy = traj.xyz[:, dims]
        ax.plot(xy[:, 0], xy[:, 1], color="#d32f2f", linewidth=0.8, alpha=0.28)

    all_points = concat_xyz(trajectories + tcp_trajectories)
    set_equal_2d(ax, all_points, dims)
    ax.grid(True, color="#eceff1", linewidth=0.7)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")


def plot_3d_view(ax: plt.Axes, trajectories: list[Trajectory], tcp_trajectories: list[Trajectory]) -> None:
    demo_trajs = [t for t in trajectories if not t.is_mapping]
    mapping = next((t for t in trajectories if t.is_mapping), None)
    cmap = matplotlib.colormaps["viridis"]
    n = max(len(demo_trajs), 1)

    lines = []
    colors = []
    for idx, traj in enumerate(demo_trajs):
        xyz = traj.xyz
        if len(xyz) < 2:
            continue
        lines.extend(np.stack([xyz[:-1], xyz[1:]], axis=1))
        colors.extend([cmap(0.15 + 0.72 * idx / max(n - 1, 1))] * (len(xyz) - 1))
    if lines:
        lc = Line3DCollection(lines, colors=colors, linewidths=0.65, alpha=0.38)
        ax.add_collection3d(lc)

    if mapping is not None:
        xyz = mapping.xyz
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#111111", linewidth=2.0, label="mapping")

    for traj in tcp_trajectories:
        xyz = traj.xyz
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#d32f2f", linewidth=0.7, alpha=0.22)

    all_points = concat_xyz(trajectories + tcp_trajectories)
    set_equal_3d(ax, all_points)
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D view", loc="left", fontsize=12, fontweight="bold")


def plot_quality_panel(ax: plt.Axes, demo_trajs: list[Trajectory], tcp_trajs: list[Trajectory], session_dir: pathlib.Path) -> None:
    ax.set_title("Tracking and dataset summary", loc="left", fontsize=12, fontweight="bold")
    if demo_trajs:
        lost_pct = np.array([100.0 * t.lost_frames / max(t.total_frames, 1) for t in demo_trajs])
        ax.hist(lost_pct, bins=min(20, max(5, len(lost_pct) // 4)), color="#607d8b", alpha=0.85)
        ax.set_xlabel("lost SLAM frames per demo (%)")
        ax.set_ylabel("demo count")
        mean_lost = lost_pct.mean()
    else:
        mean_lost = 0.0
        ax.text(0.5, 0.5, "No demo trajectories", ha="center", va="center", transform=ax.transAxes)

    plan_path = session_dir.joinpath("dataset_plan.pkl")
    if plan_path.is_file():
        plan = pickle.load(plan_path.open("rb"))
        n_episodes = len(plan)
        n_frames = sum(len(ep["episode_timestamps"]) for ep in plan)
        n_grippers = len(plan[0]["grippers"]) if n_episodes else 0
        n_cameras = len(plan[0]["cameras"]) if n_episodes else 0
    else:
        n_episodes = n_frames = n_grippers = n_cameras = 0

    summary = (
        f"Session: {session_dir.name}\n"
        f"Demo trajectories: {len(demo_trajs)}\n"
        f"Mean lost frames: {mean_lost:.2f}%\n"
        f"Planned episodes: {n_episodes}\n"
        f"Planned frames: {n_frames}\n"
        f"Grippers: {n_grippers}   Cameras: {n_cameras}\n"
        f"TCP traces drawn: {len(tcp_trajs)}"
    )
    ax.text(
        0.98,
        0.95,
        summary,
        ha="right",
        va="top",
        fontsize=9,
        color="#263238",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cfd8dc"},
    )
    ax.grid(True, axis="y", color="#eceff1")


def save_views_only_figure(
    output: pathlib.Path,
    session_dir: pathlib.Path,
    trajectories: list[Trajectory],
    tcp_trajs: list[Trajectory],
) -> None:
    fig = plt.figure(figsize=(15, 12), facecolor="#fbfcfd")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.26, wspace=0.22)

    fig.suptitle(
        f"{session_dir.name}: {'SLAM and TCP Trajectories' if tcp_trajs else 'SLAM Camera Trajectories'}",
        fontsize=17,
        fontweight="bold",
        color="#172026",
        x=0.02,
        y=0.985,
        ha="left",
    )
    subtitle = "Black: mapping camera trajectory. Colored: demo camera trajectories."
    if tcp_trajs:
        subtitle += " Red: planned TCP trajectories."
    fig.text(0.02, 0.955, subtitle, fontsize=10, color="#455a64", ha="left")

    ax_top = fig.add_subplot(gs[0, 0])
    plot_2d_view(ax_top, trajectories, tcp_trajs, dims=(0, 1), labels=("x (m)", "y (m)"), title="Top view: x-y")

    ax_side = fig.add_subplot(gs[0, 1])
    plot_2d_view(ax_side, trajectories, tcp_trajs, dims=(0, 2), labels=("x (m)", "z (m)"), title="Side view: x-z")

    ax_front = fig.add_subplot(gs[1, 0])
    plot_2d_view(ax_front, trajectories, tcp_trajs, dims=(1, 2), labels=("y (m)", "z (m)"), title="Front view: y-z")

    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")
    plot_3d_view(ax_3d, trajectories, tcp_trajs)

    legend_handles = [
        matplotlib.lines.Line2D([0], [0], color="#111111", linewidth=2.2, label="mapping camera trajectory"),
        matplotlib.lines.Line2D([0], [0], color="#4caf50", linewidth=1.4, alpha=0.7, label="demo camera trajectories"),
    ]
    if tcp_trajs:
        legend_handles.append(
            matplotlib.lines.Line2D([0], [0], color="#d32f2f", linewidth=1.4, alpha=0.7, label="planned TCP trajectories")
        )
    fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.02, 0.01), ncol=3, frameon=False)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.argument("session_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=pathlib.Path), default=None)
@click.option("--max-demos", type=int, default=None, help="Limit raw camera trajectories drawn.")
@click.option("--max-episodes", type=int, default=80, show_default=True, help="Limit planned TCP trajectories drawn.")
@click.option("--stride", type=int, default=2, show_default=True, help="Subsample trajectory points.")
@click.option("--views-only", is_flag=True, default=False, help="Save only top, side, front, and 3D views.")
def main(
    session_dir: pathlib.Path,
    output: Optional[pathlib.Path],
    max_demos: Optional[int],
    max_episodes: Optional[int],
    stride: int,
    views_only: bool,
) -> None:
    session_dir = session_dir.absolute()
    stride = max(1, stride)
    if output is None:
        output = session_dir.joinpath("slam_trajectory_mapping_overview.png")
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)

    tx_slam_tag, tx_tag_slam = load_tx_tag_slam(session_dir)
    mapping_traj = load_mapping_trajectory(session_dir, tx_tag_slam, stride)
    demo_trajs = load_demo_trajectories(session_dir, tx_tag_slam, stride, max_demos)
    tcp_trajs = load_tcp_trajectories(session_dir, stride, max_episodes)

    trajectories = []
    if mapping_traj is not None:
        trajectories.append(mapping_traj)
    trajectories.extend(demo_trajs)

    if not trajectories and not tcp_trajs:
        raise click.ClickException("No SLAM or TCP trajectories found in this session.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#b0bec5",
            "axes.labelcolor": "#263238",
            "xtick.color": "#455a64",
            "ytick.color": "#455a64",
        }
    )

    if views_only:
        save_views_only_figure(output, session_dir, trajectories, tcp_trajs)
        print(f"Saved {output}")
        print(f"Session: {session_dir}")
        print(f"Demo trajectories: {len(demo_trajs)}")
        print(f"TCP trajectories: {len(tcp_trajs)}")
        return

    fig = plt.figure(figsize=(19, 14), facecolor="#fbfcfd")
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[0.88, 1.05, 1.05], hspace=0.34, wspace=0.25)

    fig.suptitle(
        "UMI SLAM and trajectory mapping overview",
        fontsize=20,
        fontweight="bold",
        color="#172026",
        x=0.02,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.02,
        0.955,
        "Camera trajectories are shown in the table-tag frame when tx_slam_tag.json is available. "
        "Red traces are planned TCP trajectories from dataset_plan.pkl.",
        fontsize=10,
        color="#455a64",
        ha="left",
    )

    ax_pipeline = fig.add_subplot(gs[0, :2])
    draw_pipeline_panel(ax_pipeline)
    ax_transform = fig.add_subplot(gs[0, 2])
    draw_transform_panel(ax_transform, tx_slam_tag)

    ax_top = fig.add_subplot(gs[1, 0])
    plot_2d_view(ax_top, trajectories, tcp_trajs, dims=(0, 1), labels=("x (m)", "y (m)"), title="Top view: x-y")

    ax_side = fig.add_subplot(gs[1, 1])
    plot_2d_view(ax_side, trajectories, tcp_trajs, dims=(0, 2), labels=("x (m)", "z (m)"), title="Side view: x-z")

    ax_front = fig.add_subplot(gs[1, 2])
    plot_2d_view(ax_front, trajectories, tcp_trajs, dims=(1, 2), labels=("y (m)", "z (m)"), title="Front view: y-z")

    ax_3d = fig.add_subplot(gs[2, :2], projection="3d")
    plot_3d_view(ax_3d, trajectories, tcp_trajs)

    ax_quality = fig.add_subplot(gs[2, 2])
    plot_quality_panel(ax_quality, demo_trajs, tcp_trajs, session_dir)

    legend_handles = [
        matplotlib.lines.Line2D([0], [0], color="#111111", linewidth=2.2, label="mapping camera trajectory"),
        matplotlib.lines.Line2D([0], [0], color="#4caf50", linewidth=1.4, alpha=0.7, label="demo camera trajectories"),
        matplotlib.lines.Line2D([0], [0], color="#d32f2f", linewidth=1.4, alpha=0.7, label="planned TCP trajectories"),
    ]
    fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.02, 0.015), ncol=3, frameon=False)

    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")
    print(f"Session: {session_dir}")
    print(f"Demo trajectories: {len(demo_trajs)}")
    print(f"TCP trajectories: {len(tcp_trajs)}")


if __name__ == "__main__":
    main()
