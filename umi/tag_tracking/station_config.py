"""
Station setup from what the videos show (scripts_tag_tracking/propose_station_config.py):
finger tags, cubes, which cube is on which gripper, environment and dock tags.
Input is per-frame tag detections (tag_detection.sample_detections) and their
tag_inventory; tag classes come from the tag sizes in the aruco config.
"""
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


def tag_classes(sizes: Dict[int, float], finger_size: float, cube_size: float) -> Dict[str, Set[int]]:
    """
    Configured tag ids by size: finger tags, cube tags, environment tags (every other size).
    sizes: tag id -> side [m], only the ids listed in the aruco config.
    """
    out = {'finger': set(), 'cube': set(), 'environment': set()}
    for tag_id, size in sizes.items():
        if abs(size - finger_size) < 1e-6:
            out['finger'].add(int(tag_id))
        elif abs(size - cube_size) < 1e-6:
            out['cube'].add(int(tag_id))
        else:
            out['environment'].add(int(tag_id))
    return out


def finger_candidates(inventory: Dict[int, dict], finger_ids: Set[int], min_share: float = 0.5,
        min_y: float = 0.55) -> List[int]:
    """
    Finger-size tags of a wrist video in at least min_share of the frames, in the lower
    part of the image (y >= min_y).
    """
    return [t for t, x in inventory.items() if t in finger_ids and x['share'] >= min_share
        and x['center'][1] >= min_y]


def finger_pair(inventory: Dict[int, dict], finger_ids: Set[int], min_share: float = 0.5,
        min_y: float = 0.55) -> Optional[Tuple[int, int]]:
    """
    (image-left, image-right) finger tag of a wrist video: two finger_candidates, one
    left and one right of the center, mounted symmetrically: the pair whose median
    positions mirror each other best about the vertical image center line (another
    gripper's finger tag, seen at the image edge, does not). None if there is none.
    """
    cands = finger_candidates(inventory, finger_ids, min_share, min_y)
    left = [t for t in cands if inventory[t]['center'][0] < 0.5]
    right = [t for t in cands if inventory[t]['center'][0] >= 0.5]
    pairs = [(l, r) for l in left for r in right]
    if len(pairs) == 0:
        return None
    return min(pairs, key=lambda p: finger_asymmetry(inventory, p))


def finger_asymmetry(inventory: Dict[int, dict], pair: Tuple[int, int]) -> float:
    """
    How far two tags are from mirror positions about the vertical center line (image width units).
    """
    (xl, yl), (xr, yr) = inventory[pair[0]]['center'], inventory[pair[1]]['center']
    return abs(xl + xr - 1.0) + abs(yl - yr)


def _center_side(c: np.ndarray) -> Tuple[np.ndarray, float]:
    return c.mean(axis=0), float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))


def cube_groups(detections: Sequence[Dict[int, np.ndarray]], cube_ids: Set[int],
        max_ratio: float = 2.0, min_frames: int = 3) -> Tuple[List[List[int]], Dict[Tuple[int, int], dict]]:
    """
    Cube-size tags grouped into cubes: two tags are on one cube if, over the frames
    showing both (at least min_frames), the median distance of their centers is below
    max_ratio tag side lengths (tags on one cube are about one side apart, tags of
    two cubes held in two hands much farther). Groups are the connected pairs, so
    opposite faces that are never seen together still join through a side face.
    Returns (groups sorted by smallest id, per pair {'frames', 'ratio'}).
    """
    ratios: Dict[Tuple[int, int], list] = dict()
    seen = set()
    for corners in detections:
        tags = sorted(t for t in corners if t in cube_ids)
        seen.update(tags)
        cs = {t: _center_side(corners[t]) for t in tags}
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                d = np.linalg.norm(cs[a][0] - cs[b][0]) / (0.5 * (cs[a][1] + cs[b][1]))
                ratios.setdefault((a, b), list()).append(d)
    pairs = {k: {'frames': len(v), 'ratio': float(np.median(v))} for k, v in ratios.items()}
    parent = {t: t for t in seen}

    def root(t):
        while parent[t] != t:
            t = parent[t]
        return t
    for (a, b), x in pairs.items():
        if x['frames'] >= min_frames and x['ratio'] < max_ratio:
            parent[root(b)] = root(a)
    groups = dict()
    for t in seen:
        groups.setdefault(root(t), list()).append(t)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0]), pairs


def assign_cubes(cubes: List[List[int]], wrist_inventories: Dict[str, Dict[int, dict]]
        ) -> Tuple[Optional[Dict[str, int]], Dict[str, List[float]]]:
    """
    Which cube is on which of the two grippers: a wrist camera does not see the cube
    on its own gripper, it sees the other gripper's cube when the hands cross.
    wrist_inventories: role -> tag_inventory of that wrist camera's take videos.
    Returns (role -> index into cubes, or None if neither wrist sees a cube;
    role -> per cube the summed share of frames its tags are seen in).
    """
    roles = sorted(wrist_inventories)
    assert len(roles) == 2 and len(cubes) == 2
    seen = {r: [sum(wrist_inventories[r].get(t, {'share': 0.0})['share'] for t in c) for c in cubes]
        for r in roles}
    a, b = roles
    # a's own cube is the one b sees
    straight = seen[b][0] + seen[a][1]   # a: cube 0, b: cube 1
    crossed = seen[b][1] + seen[a][0]    # a: cube 1, b: cube 0
    if straight == crossed:
        return None, seen
    return ({a: 0, b: 1} if straight > crossed else {a: 1, b: 0}), seen


def frame_positions(corners: Dict[int, np.ndarray], cubes: List[List[int]], environment_ids: Set[int],
        tag_sizes: Dict[int, float], camera) -> Tuple[List[Optional[np.ndarray]], Dict[int, np.ndarray]]:
    """
    3D positions in the head camera of one frame, as wrist_events.docked_view: per cube
    its center (cube half size behind its unambiguous visible tags, median) or None, and
    per visible environment tag its position.
    """
    from umi.tag_tracking.initialization import unambiguous_tag_pose
    from umi.tag_tracking.wrist_events import DOCK_POSE_SIGMA_PX, UMI_CUBE_HALF_SIZE
    pose = lambda t: unambiguous_tag_pose(camera, corners[t], tag_sizes[t], DOCK_POSE_SIGMA_PX)
    centers = list()
    for cube in cubes:
        c = [T[:3, 3] - UMI_CUBE_HALF_SIZE * T[:3, 2] for T in (pose(t) for t in cube if t in corners)
            if T is not None]
        centers.append(np.median(c, axis=0) if len(c) else None)
    env = {t: T[:3, 3] for t, T in ((t, pose(t)) for t in corners if t in environment_ids) if T is not None}
    return centers, env


def dock_tags(videos: Sequence[Sequence[Tuple[List[Optional[np.ndarray]], Dict[int, np.ndarray]]]],
        dock_distance: float = 0.35, max_move: float = 0.03) -> Tuple[List[dict], int]:
    """
    The environment tags next to the docked grippers. Docked frame: both cubes within
    dock_distance (m) of an environment tag (wrist_events.docked_view) and both still in
    the world since the previous sampled frame of the same video: their distances to
    the environment tags seen in both frames changed by less than max_move (m); in a
    task at most one hand rests. Per cube the closest environment tag is voted over the
    docked frames. videos: per head video its frame_positions per sampled frame.
    Returns (per cube [{'tag', 'share', 'distance' (median, m)}, most votes first],
    number of docked frames).
    """
    def dists(f):
        centers, env = f
        if any(c is None for c in centers) or len(env) == 0:
            return None
        return [{t: float(np.linalg.norm(c - p)) for t, p in env.items()} for c in centers]

    votes = None
    n = 0
    for frames in videos:
        prev = None
        for f in frames:
            cur = dists(f)
            if votes is None and cur is not None:
                votes = [dict() for _ in cur]
            if cur is not None and prev is not None and all(min(d.values()) <= dock_distance for d in cur):
                common = set(cur[0]) & set(prev[0])
                if len(common) > 0 and all(abs(c[t] - p[t]) < max_move for c, p in zip(cur, prev) for t in common):
                    n += 1
                    for v, d in zip(votes, cur):
                        t = min(d, key=d.get)
                        v.setdefault(t, list()).append(d[t])
            prev = cur
    if votes is None or n == 0:
        return [], 0
    return [[{'tag': t, 'share': len(v[t]) / n, 'distance': float(np.median(v[t]))}
        for t in sorted(v, key=lambda t: -len(v[t]))] for v in votes], n


def finger_opening(detections: Sequence[Dict[int, np.ndarray]], image_size: Sequence[int],
        pair: Optional[Tuple[int, int]]) -> Dict[str, np.ndarray]:
    """
    Per sampled wrist frame the finger opening (image width units) of the video's finger
    tag pair (finger_pair of its tag_inventory, fixed for the video: picking a pair per
    frame would jump to another gripper's finger tag whenever one is not detected), as
    three signals, NaN where not detected:
    'pair': distance between the two tags; 'left' / 'right': twice one tag's distance
    from the image center (a finger tag may leave the view or blur when open, the
    other one still shows the motion). All NaN if pair is None.
    """
    n = len(detections)
    out = {k: np.full(n, np.nan) for k in ['pair', 'left', 'right']}
    if pair is None:
        return out
    l, r = pair
    w = float(image_size[0])
    for i, corners in enumerate(detections):
        xl = corners[l][:, 0].mean() / w if l in corners else np.nan
        xr = corners[r][:, 0].mean() / w if r in corners else np.nan
        out['pair'][i] = xr - xl
        out['left'][i] = 2 * (0.5 - xl)
        out['right'][i] = 2 * (xr - 0.5)
    return out


def open_close_cycles(opening: np.ndarray, low: float = 0.3, high: float = 0.7) -> Tuple[int, float]:
    """
    Full open-close cycles of a finger opening series: from below low to above high
    of its own 5-95 percentile range and back. Returns (cycles, range in image width units).
    """
    x = opening[np.isfinite(opening)]
    if len(x) < 2:
        return 0, 0.0
    lo, hi = np.percentile(x, [5, 95])
    if hi - lo <= 0:
        return 0, 0.0
    u = (x - lo) / (hi - lo)
    cycles, state = 0, None
    for v in u:
        if v <= low:
            if state == 'open':
                cycles += 1
            state = 'closed'
        elif v >= high:
            state = 'open'
    return cycles, float(hi - lo)


def clip_role(head_inventory: Dict[int, dict], wrist_openings: Dict[str, Dict[str, np.ndarray]],
        environment_ids: Set[int], min_cycles: int = 2, min_range: float = 0.08, min_seen: float = 0.5,
        min_env_tags: int = 3, min_share: float = 0.05) -> Tuple[str, List[str], str]:
    """
    What a short recording is, from what its cameras show. wrist_openings: per wrist
    its finger_opening signals; a signal counts if seen in >= min_seen of the frames.
    - 'gripper_calibration': a wrist's fingers open and close fully at least min_cycles
      times over at least min_range of the image width in one of its signals (that
      wrist calibrated);
    - 'head_mapping': otherwise, if the head sees at least min_env_tags environment tags
      (in >= min_share of its frames); a finger moved once during the walk does not
      matter (station1 2026-11-20 recording 35);
    - 'unknown' otherwise.
    Returns (role, calibrated wrists, evidence text).
    """
    best, evidence = dict(), list()
    for w, signals in sorted(wrist_openings.items()):
        stats = {k: open_close_cycles(x) + (float(np.mean(np.isfinite(x))) if len(x) else 0.0,)
            for k, x in signals.items()}
        seen = {k: v for k, v in stats.items() if v[2] >= min_seen}
        if seen:
            k = max(seen, key=lambda k: (seen[k][0] if seen[k][1] >= min_range else -1, seen[k][1]))
            best[w] = seen[k]
            evidence.append(f"{w} fingers ({k}) {seen[k][0]} cycles over {seen[k][1]:.2f}, seen {seen[k][2]:.0%}")
        else:
            evidence.append(f"{w} fingers seen in < {min_seen:.0%} of the frames")
    env = sorted(t for t, x in head_inventory.items() if t in environment_ids and x['share'] >= min_share)
    evidence = '; '.join(evidence + [f"head {len(env)} environment tags"])
    calibrated = sorted(w for w, (c, r, _) in best.items() if c >= min_cycles and r >= min_range)
    if calibrated:
        return 'gripper_calibration', calibrated, evidence
    if len(env) >= min_env_tags:
        return 'head_mapping', [], evidence
    return 'unknown', [], evidence
