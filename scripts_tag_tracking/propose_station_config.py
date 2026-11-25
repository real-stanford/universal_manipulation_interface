"""
Draft tag tracking and gripper configs of a new station from one sorted session
(raw_videos/ with pairs.csv, gripper_calibration/, head_mapping/, from
sort_recordings.py), instead of reading inspect_tags.py output by hand.

What is read from the videos (tag classes by size in the aruco config):
- camera serials of head / left / right (GoPro metadata), one serial per role;
- finger tags per wrist camera: in the gripper calibration video the two
  finger-size tags near the bottom, one left and one right of the center, at
  mirror positions (no such pair: not a wrist camera, stop);
- cubes: cube-size tags seen by the head in the takes, grouped by how close
  they stay to each other (umi.tag_tracking.station_config.cube_groups);
  cube-size tags never seen are reported, not added;
- which cube is on which gripper: a wrist camera sees the other gripper's
  cube, never its own;
- environment tags: environment-size tags in the head mapping video;
- dock tags (with --head_intrinsics): head frames where both cubes are still in
  the world and within episodes.dock_distance of an environment tag (3D from
  single tags, as wrist_events.docked_view); per cube its closest environment
  tag, kept if closest in >= --dock_share of those frames.
UMI order: right gripper robot_index / gripper_id 0, left 1.
Gripper geometry is not in the videos: TCP --tcp_mm in front of the lens
(default 192 mm, the measured 2026-11-16 grippers), the rest as in
gripper_config_station1.yaml, all overridable.
Not filled (null, from other steps): cameras (OpenICC intrinsics per serial),
head_imu / gripper_imu (import_openicc_calibration.py), setup_calibration
(calibrate_setup.py). Everything else is copied from --template.

python scripts_tag_tracking/propose_station_config.py <session> -n <station> \\
    [-a example/tag_tracking/aruco_config.yaml] [-t example/tag_tracking/tag_tracking_station1.yaml] \\
    [-o example/tag_tracking]
writes <output>/tag_tracking_<station>.yaml and gripper_config_<station>.yaml.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import concurrent.futures
import csv
import multiprocessing
import pathlib
import time
import click
import yaml
from umi.common.exiftool_util import ExifTool
from umi.tag_tracking.tag_detection import sample_detections, tag_inventory
from umi.tag_tracking.station_config import tag_classes, finger_candidates, finger_pair, finger_asymmetry, cube_groups, assign_cubes, \
    frame_positions, dock_tags
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.config import EPISODE_DEFAULTS

WRISTS = {'right': 0, 'left': 1}   # role -> robot_index / gripper_id (UMI convention)
CUBE_PAIR_MIN_FRAMES = 3           # two cube tags grouped only if seen together in at least this many frames


def session_videos(raw: pathlib.Path) -> dict:
    """
    Videos per role from raw_videos/pairs.csv: takes (head, left, right per recording),
    gripper calibration per wrist role, head mapping.
    """
    rows = list(csv.DictReader(open(raw.joinpath('pairs.csv'))))
    out = {'takes': list(), 'gripper_calibration': dict(), 'head_mapping': list()}
    for r in rows:
        if r['type'] == 'take':
            out['takes'].append({role: raw.joinpath(r[role]) for role in ['head', *WRISTS] if r[role]})
        elif r['type'] == 'gripper_calibration':
            role = [x for x in WRISTS if r[x]]
            assert len(role) == 1, f"pairs.csv gripper_calibration row without one wrist: {r}"
            out['gripper_calibration'][role[0]] = raw.joinpath(r['note'] or r[role[0]])
        elif r['type'] == 'head_mapping':
            out['head_mapping'].append(raw.joinpath(r['note'] or r['head']))
    for path in [p for t in out['takes'] for p in t.values()] + list(out['gripper_calibration'].values()) \
            + out['head_mapping']:
        assert path.is_file(), f"{path} in pairs.csv is missing"
    assert len(out['takes']) > 0 and len(out['head_mapping']) > 0, "need takes and a head mapping video"
    assert set(out['gripper_calibration']) == set(WRISTS), "need a gripper calibration video per wrist"
    return out


class Report(list):
    """
    Lines for the config headers, printed as they come (so a stop still shows the evidence).
    """
    def append(self, line):
        print(line, flush=True)
        super().append(line)


def merged_inventory(results: list) -> dict:
    detections, size = list(), None
    for r in results:
        detections.extend(r['detections'])
        size = r['size']
    return tag_inventory(detections, size)


# %%
@click.command()
@click.argument('session')
@click.option('-n', '--name', required=True, help='Station name, output tag_tracking_<name>.yaml')
@click.option('-a', '--aruco_config', default='example/tag_tracking/aruco_config.yaml')
@click.option('-t', '--template', default='example/tag_tracking/tag_tracking_station1.yaml',
    help='Tag tracking config whose other settings (accuracy, episodes, ...) are copied')
@click.option('-o', '--output', default='example/tag_tracking', help='Output directory')
@click.option('--finger_size', type=float, default=0.016, help='Finger tag side [m]')
@click.option('--cube_size', type=float, default=0.06, help='Cube tag side [m]')
@click.option('--tcp_mm', type=float, default=192.0, help='Fingertips in front of the lens [mm]')
@click.option('--cam_to_mount_offset', type=float, default=0.01465, help='Lens to mount [m]')
@click.option('--cam_to_center_height', type=float, default=0.075, help='Fingertips below the lens center [m]')
@click.option('--nominal_z', type=float, default=0.079, help='Finger tags in front of the lens [m]')
@click.option('--finger_tag_to_tip', type=float, default=0.113, help='Finger tag centers to fingertips [m]')
@click.option('--every_sec', type=float, default=1.0, help='Sample one frame every this many seconds')
@click.option('--max_sec', type=float, default=180.0, help='Only the first this many seconds of each take video')
@click.option('--min_share', type=float, default=0.05, help='Environment tags in at least this share of mapping frames')
@click.option('--dock_share', type=float, default=0.25, help='Dock tag: closest to a resting cube in this share of its frames')
@click.option('--head_intrinsics', default=None, help='OpenICC intrinsics json of the head camera, for the dock tags')
@click.option('-j', '--jobs', type=int, default=8, help='Videos read in parallel')
@click.option('--overwrite', is_flag=True, help='Replace existing output files')
def main(session, name, aruco_config, template, output, finger_size, cube_size, tcp_mm, cam_to_mount_offset,
        cam_to_center_height, nominal_z, finger_tag_to_tip, every_sec, max_sec, min_share, dock_share, head_intrinsics, jobs, overwrite):
    raw = pathlib.Path(os.path.expanduser(session)).absolute().joinpath('raw_videos')
    out_dir = pathlib.Path(os.path.expanduser(output)).absolute()
    tt_path = out_dir.joinpath(f'tag_tracking_{name}.yaml')
    gc_path = out_dir.joinpath(f'gripper_config_{name}.yaml')
    for p in [tt_path, gc_path]:
        assert overwrite or not p.exists(), f"{p} exists (--overwrite to replace)"
    aruco = yaml.safe_load(open(os.path.expanduser(aruco_config), 'r'))
    predefined = aruco['aruco_dict']['predefined']
    sizes = {int(k): float(v) for k, v in aruco['marker_size_map'].items() if k != 'default'}
    classes = tag_classes(sizes, finger_size, cube_size)
    videos = session_videos(raw)

    # camera serial per role, one per role
    by_role = {'head': [t['head'] for t in videos['takes'] if 'head' in t] + videos['head_mapping']}
    for role in WRISTS:
        by_role[role] = [t[role] for t in videos['takes'] if role in t] + [videos['gripper_calibration'][role]]
    print(f"{len(videos['takes'])} takes, gripper calibration and head mapping videos in {raw}; reading camera serials",
        flush=True)
    with ExifTool() as et:
        serial_of = {p: str(list(et.get_metadata(str(p)))[0].get('QuickTime:CameraSerialNumber'))
            for paths in by_role.values() for p in paths}
    serials = dict()
    for role, paths in by_role.items():
        s = {serial_of[p] for p in paths}
        assert len(s) == 1, f"{role}: videos from several cameras: " + \
            ', '.join(f"{p.name} {serial_of[p]}" for p in paths)
        serials[role] = s.pop()
    assert len(set(serials.values())) == 3, f"one camera in two roles: {serials}"

    # sample tags
    jobs_in = [(p, None) for p in videos['head_mapping']] + \
        [(videos['gripper_calibration'][r], None) for r in WRISTS] + \
        [(p, max_sec) for t in videos['takes'] for p in t.values()]
    print(f"reading tags in {len(jobs_in)} videos (a frame every {every_sec:g} s; takes: the first {max_sec:g} s), "
        f"{min(jobs, len(jobs_in))} at a time", flush=True)
    start = time.time()
    results = dict()
    # spawn, not fork: a forked worker can inherit a held lock (decoder / OpenCV threads) and hang idle
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(sample_detections, p, predefined, every_sec, m): p for p, m in jobs_in}
        for k, f in enumerate(concurrent.futures.as_completed(futures), start=1):
            p = futures[f]
            results[p] = f.result()
            print(f"  {k}/{len(jobs_in)} {p.parent.name}/{p.name} ({time.time() - start:.0f} s)", flush=True)
    report = Report()
    report.append(f"cameras: head {serials['head']}, right {serials['right']}, left {serials['left']}")

    # finger tags (a wrist camera may also see the other gripper's fingers, off the mirror positions)
    inv = {r: tag_inventory(results[videos['gripper_calibration'][r]]['detections'],
        results[videos['gripper_calibration'][r]]['size']) for r in WRISTS}
    fingers = {r: finger_pair(inv[r], classes['finger']) for r in WRISTS}
    for r in WRISTS:
        cands = finger_candidates(inv[r], classes['finger'])
        report.append(f"{r} camera {serials[r]}, {videos['gripper_calibration'][r].name}: finger candidates "
            + ', '.join(f"{t} ({inv[r][t]['share']:.0%}, x {inv[r][t]['center'][0]:.2f})" for t in cands))
        assert fingers[r] is not None, f"{r}: no finger tag pair; not a wrist camera or other finger tag size"
        report.append(f"{r} finger pair asymmetry {finger_asymmetry(inv[r], fingers[r]):.2f} (0 = mirror positions)")
    if set(fingers['right']) == set(fingers['left']):
        report.append(f"note: both grippers have finger tags {sorted(fingers['right'])} (allowed, "
            "each is only looked for in its own wrist camera)")
    for r in WRISTS:
        report.append(f"{r} finger tags {fingers[r][0]} (image left) / {fingers[r][1]} (image right)")

    # cubes from the head takes
    head_det = [d for t in videos['takes'] if 'head' in t for d in results[t['head']]['detections']]
    cubes, pairs = cube_groups(head_det, classes['cube'], min_frames=CUBE_PAIR_MIN_FRAMES)
    for (a, b), x in sorted(pairs.items()):
        report.append(f"head takes: cube tags {a}-{b} seen together in {x['frames']} frames, "
            f"{x['ratio']:.1f} sides apart")
    # a tag alone in its group, seen with no other tag in enough frames to group it (a face mostly
    # turned away): not a cube of its own, reported like the tags never seen
    rare = [g[0] for g in cubes if len(g) == 1
        and all(x['frames'] < CUBE_PAIR_MIN_FRAMES for p, x in pairs.items() if g[0] in p)]
    cubes = [g for g in cubes if not (len(g) == 1 and g[0] in rare)]
    if rare:
        report.append(f"WARNING cube-size tags seen by the head too rarely to group (with another tag in "
            f"< {CUBE_PAIR_MIN_FRAMES} frames): {rare}; left out, add them to their cube if they belong to it")
    assert len(cubes) == 2, f"head takes: {len(cubes)} cubes {cubes}, expected 2; " \
        "check the pair distances above (two cubes always held together, or a tag never seen with another)"
    unseen = sorted(classes['cube'] - {t for c in cubes for t in c} - set(rare))
    if unseen:
        report.append(f"WARNING cube-size tags in the aruco config never seen by the head: {unseen} "
            "(a cube face turned away, or not in this station); add them to their cube if they belong to it")
    wrist_inv = {r: merged_inventory([results[t[r]] for t in videos['takes'] if r in t]) for r in WRISTS}
    owner, seen = assign_cubes(cubes, wrist_inv)
    for r in WRISTS:
        report.append(f"{r} wrist takes: sees cube {cubes[0]} {seen[r][0]:.0%}, cube {cubes[1]} {seen[r][1]:.0%} "
            "(summed share of frames per tag)")
    assert owner is not None, "no wrist camera sees a cube: cube to gripper not decided"
    weak = min(seen[r][1 - owner[r]] for r in WRISTS)
    report.append(f"cube {cubes[owner['right']]} on the right gripper, {cubes[owner['left']]} on the left"
        + (" (one wrist never sees the other cube: from one camera only)" if weak == 0 else ''))

    # environment
    mapping_inv = merged_inventory([results[p] for p in videos['head_mapping']])
    environment = sorted(t for t, x in mapping_inv.items() if t in classes['environment'] and x['share'] >= min_share)
    head_inv = merged_inventory([results[t['head']] for t in videos['takes'] if 'head' in t])
    report.append(f"head mapping: environment tags {environment}")
    missing = sorted(t for t, x in head_inv.items() if t in classes['environment'] and t not in environment
        and x['share'] >= min_share)
    if missing:
        report.append(f"WARNING head takes show environment-size tags not in the mapping: {missing}")
    unlisted = sorted(t for inv in [mapping_inv, head_inv] for t, x in inv.items() if t not in sizes
        and x['share'] >= min_share)
    if unlisted:
        report.append(f"WARNING tags without a size in the aruco config (ignored): {sorted(set(unlisted))}")

    # dock tags: where both cubes rest, in 3D as the pipeline's docked view
    config = yaml.safe_load(open(os.path.expanduser(template), 'r'))
    dock = list()
    if head_intrinsics is None:
        report.append("dock tags not proposed: needs --head_intrinsics (3D distances); pick them from "
            "the environment tags next to the dock")
    else:
        camera = FisheyeCamera(os.path.expanduser(head_intrinsics))
        dock_distance = config.get('episodes', {}).get('dock_distance', EPISODE_DEFAULTS['dock_distance'])
        positions = [[frame_positions(c, cubes, set(environment), sizes, camera)
            for c in results[t['head']]['detections']] for t in videos['takes'] if 'head' in t]
        per_cube, n = dock_tags(positions, dock_distance)
        report.append(f"docked: {n} head frames with both cubes resting within {dock_distance} m of an "
            "environment tag")
        for role in WRISTS:
            if n == 0:
                break
            tags = per_cube[owner[role]]
            dock.extend(x['tag'] for x in tags if x['share'] >= dock_share)
            report.append(f"{role} cube docked, closest environment tag: " + ', '.join(
                f"{x['tag']} {x['share']:.0%} ({x['distance']:.2f} m)" for x in tags[:3]))
        dock = sorted(set(dock))
        report.append(f"dock tags {dock} (closest to a docked cube in >= {dock_share:.0%} of the docked frames)")

    # tag tracking config
    config['gripper_config'] = gc_path.name
    config['head_camera'] = serials['head']
    config['setup_calibration'] = f'setup_{name}'
    config['head_imu'] = None
    config['gripper_imu'] = {serials[r]: None for r in WRISTS}
    config['cameras'] = {serials[r]: None for r in ['head', *WRISTS]}
    if head_intrinsics is not None:
        config['cameras'][serials['head']] = os.path.relpath(os.path.abspath(os.path.expanduser(head_intrinsics)), out_dir)
    config['bodies'] = {'environment': {'static': True, 'tags': environment}}
    for role, index in WRISTS.items():
        config['bodies'][f'{role}_cube'] = {'tags': cubes[owner[role]], 'gripper_camera': serials[role],
            'robot_index': index}
    if 'episodes' in config:
        config['episodes']['dock_tags'] = dock
    header = [f"Tag tracking config for {name}, drafted by scripts_tag_tracking/propose_station_config.py",
        f"from session {raw.parent.name} (template {os.path.basename(template)}).",
        "To fill: cameras (OpenICC intrinsics json per serial), head_imu / gripper_imu",
        f"(import_openicc_calibration.py), setup_{name}/ (calibrate_setup.py). Check dock_tags.", ''] + report
    tt_path.parent.mkdir(parents=True, exist_ok=True)
    tt_path.write_text(''.join(f"# {l}\n" if l else "#\n" for l in header) + yaml.safe_dump(config, sort_keys=False))

    # gripper config
    tcp_offset = round(tcp_mm / 1000 - cam_to_mount_offset, 6)
    grippers = {serials[r]: {'gripper_id': i, 'left_finger_tag_id': fingers[r][0],
        'right_finger_tag_id': fingers[r][1], 'finger_tag_to_tip': finger_tag_to_tip} for r, i in WRISTS.items()}
    gconf = {'nominal_z': nominal_z, 'cam_to_center_height': cam_to_center_height,
        'cam_to_mount_offset': cam_to_mount_offset, 'tcp_offset': tcp_offset, 'grippers': grippers}
    header = [f"Gripper config for {name}, drafted by scripts_tag_tracking/propose_station_config.py.",
        "Finger tags (image left / right) from the gripper calibration videos;",
        f"TCP {tcp_mm:.0f} mm in front of the lens (tcp_offset + cam_to_mount_offset), "
        f"{cam_to_center_height * 1000:.0f} mm below its center: not measured for these grippers.",
        *[f"{r} hand: camera {serials[r]}" for r in WRISTS]]
    gc_path.write_text(''.join(f"# {l}\n" for l in header) + yaml.safe_dump(gconf, sort_keys=False))

    print(f"\nwrote {tt_path}\n      {gc_path}")

# %%
if __name__ == "__main__":
    main()
