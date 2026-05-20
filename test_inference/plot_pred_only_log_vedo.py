"""Vedo plot for prediction-only deployment JSONL logs.

`eval_pred_only.py` writes one JSON object per policy call. Each record stores
the current robot pose and the full absolute action horizon using this layout:

    [x, y, z, rx, ry, rz, gripper_width]

where `rx, ry, rz` is an axis-angle / rotation-vector orientation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from scipy.spatial.transform import Rotation as R


DEFAULT_LOG_PATH = Path("test_inference/pred_only_log_20260520_154521.jsonl")


def load_jsonl_record(log_path: str | Path, line_index: int = 0) -> dict:
    """Load one JSONL record, defaulting to the first policy-call scene."""
    log_path = Path(log_path)
    with log_path.open("r") as f:
        for idx, line in enumerate(f):
            if idx == line_index:
                return json.loads(line)
    raise IndexError(f"{log_path} has no line_index={line_index}.")


def reshape_horizon_by_robot(full_horizon_prediction: np.ndarray, num_robots: int) -> np.ndarray:
    """Return full horizon as (horizon, num_robots, 7)."""
    full_horizon_prediction = np.asarray(full_horizon_prediction, dtype=np.float64)
    if full_horizon_prediction.ndim != 2:
        raise ValueError(
            "full_horizon_prediction must have shape (horizon, 7 * num_robots); "
            f"got {full_horizon_prediction.shape}."
        )

    expected_width = 7 * num_robots
    if full_horizon_prediction.shape[1] != expected_width:
        raise ValueError(
            "full_horizon_prediction width does not match current_pose robot count: "
            f"got width={full_horizon_prediction.shape[1]}, expected={expected_width}."
        )

    return full_horizon_prediction.reshape(
        full_horizon_prediction.shape[0],
        num_robots,
        7,
    )


def _make_line(vedo, points, color, linewidth=2, alpha=1.0):
    try:
        return vedo.Line(points, c=color, lw=linewidth, alpha=alpha)
    except TypeError:
        line = vedo.Line(points, c=color, lw=linewidth)
        if hasattr(line, "alpha"):
            line.alpha(alpha)
        return line


def _make_points(vedo, points, color, radius=8, alpha=1.0):
    try:
        return vedo.Points(points, c=color, r=radius, alpha=alpha)
    except TypeError:
        points_actor = vedo.Points(points, c=color, r=radius)
        if hasattr(points_actor, "alpha"):
            points_actor.alpha(alpha)
        return points_actor


def _make_arrows(vedo, starts, ends, color, alpha=1.0):
    try:
        return vedo.Arrows(starts, ends, c=color, alpha=alpha)
    except TypeError:
        arrows = vedo.Arrows(starts, ends, c=color)
        if hasattr(arrows, "alpha"):
            arrows.alpha(alpha)
        return arrows


def _make_gradient_line(vedo, points, cmap_name="Oranges", linewidth=5, alpha=0.95):
    cmap = plt.get_cmap(cmap_name)
    num_segments = max(len(points) - 1, 1)
    color_values = np.linspace(0.95, 0.35, num_segments)

    actors = []
    for segment_idx, color_value in enumerate(color_values):
        segment_points = points[segment_idx : segment_idx + 2]
        segment_color = to_hex(cmap(color_value))
        actors.append(_make_line(vedo, segment_points, segment_color, linewidth, alpha))
    return actors


def _axis_arrow_actors(
    vedo,
    xyz: np.ndarray,
    rotvec: np.ndarray,
    axis_scale: float,
    alpha: float,
):
    xyz = np.asarray(xyz, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    rotations = R.from_rotvec(rotvec).as_matrix()

    x_ends = xyz + rotations[:, :, 0] * axis_scale
    y_ends = xyz + rotations[:, :, 1] * axis_scale
    z_ends = xyz + rotations[:, :, 2] * axis_scale

    return [
        _make_arrows(vedo, xyz, x_ends, "red", alpha=alpha),
        _make_arrows(vedo, xyz, y_ends, "green", alpha=alpha),
        _make_arrows(vedo, xyz, z_ends, "blue", alpha=alpha),
    ]


def build_pred_only_vedo_actors(
    record: dict,
    robot_idx: int = 0,
    axis_scale: float = 0.025,
    num_rotation_markers: int = 16,
):
    import vedo

    current_pose = np.asarray(record["current_pose"], dtype=np.float64)
    if current_pose.ndim != 2 or current_pose.shape[1] != 7:
        raise ValueError(f"current_pose must have shape (num_robots, 7); got {current_pose.shape}.")

    if robot_idx < 0 or robot_idx >= current_pose.shape[0]:
        raise ValueError(f"robot_idx={robot_idx} is outside num_robots={current_pose.shape[0]}.")

    horizon_by_robot = reshape_horizon_by_robot(
        np.asarray(record["full_horizon_prediction"], dtype=np.float64),
        num_robots=current_pose.shape[0],
    )

    current = current_pose[robot_idx]
    prediction = horizon_by_robot[:, robot_idx, :]

    current_xyz = current[None, :3]
    current_rotvec = current[None, 3:6]
    predicted_xyz = prediction[:, :3]
    predicted_rotvec = prediction[:, 3:6]
    trajectory_xyz = np.vstack([current_xyz, predicted_xyz])

    marker_count = min(num_rotation_markers, len(predicted_xyz))
    marker_positions = np.unique(
        np.linspace(0, len(predicted_xyz) - 1, marker_count, dtype=int)
    )
    predicted_marker_xyz = predicted_xyz[marker_positions]
    predicted_marker_rotvec = predicted_rotvec[marker_positions]

    actors = [
        _make_points(vedo, current_xyz, "yellow", radius=18, alpha=1.0),
        _make_points(vedo, predicted_xyz, "#1f77b4", radius=8, alpha=0.95),
    ]
    actors.extend(_make_gradient_line(vedo, trajectory_xyz, linewidth=5, alpha=0.95))
    actors.extend(
        _axis_arrow_actors(
            vedo,
            current_xyz,
            current_rotvec,
            axis_scale=axis_scale * 1.35,
            alpha=1.0,
        )
    )
    actors.extend(
        _axis_arrow_actors(
            vedo,
            predicted_marker_xyz,
            predicted_marker_rotvec,
            axis_scale=axis_scale,
            alpha=0.8,
        )
    )

    return actors


def plot_pred_only_jsonl(
    log_path: str | Path = DEFAULT_LOG_PATH,
    line_index: int = 5,
    robot_idx: int = 0,
    axis_scale: float = 0.025,
    num_rotation_markers: int = 16,
    show: bool = True,
):
    """Load one pred-only JSONL line and show its current/predicted pose horizon."""
    import vedo

    record = load_jsonl_record(log_path, line_index=line_index)
    actors = build_pred_only_vedo_actors(
        record,
        robot_idx=robot_idx,
        axis_scale=axis_scale,
        num_rotation_markers=num_rotation_markers,
    )

    current_pose = np.asarray(record["current_pose"], dtype=np.float64)
    horizon = np.asarray(record["full_horizon_prediction"], dtype=np.float64)

    print("log file:", log_path)
    print("line index:", line_index)
    print("record log_idx:", record.get("log_idx"))
    print("robot index:", robot_idx)
    print("current pose [x y z rx ry rz gripper]:")
    print(np.round(current_pose[robot_idx], 5))
    print("prediction horizon shape:", horizon.shape)
    print("num_actions_that_would_execute:", record.get("num_actions_that_would_execute"))
    print("yellow dot + bright axes = current pose")
    print("orange gradient line = predicted position horizon")
    print("blue dots + red/green/blue axes = predicted poses and rotations")

    if not show:
        return record, actors

    vedo.settings.default_backend = "vtk"
    return vedo.show(
        actors,
        axes=1,
        viewup="z",
        bg="white",
        title=f"Pred-only deployment horizon: log_idx={record.get('log_idx')}, robot={robot_idx}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        nargs="?",
        default=str(DEFAULT_LOG_PATH),
        help="Prediction-only JSONL log path.",
    )
    parser.add_argument("--line-index", type=int, default=0)
    parser.add_argument("--robot-idx", type=int, default=0)
    parser.add_argument("--axis-scale", type=float, default=0.025)
    parser.add_argument("--num-rotation-markers", type=int, default=16)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plot_pred_only_jsonl(
        args.log_path,
        line_index=args.line_index,
        robot_idx=args.robot_idx,
        axis_scale=args.axis_scale,
        num_rotation_markers=args.num_rotation_markers,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
