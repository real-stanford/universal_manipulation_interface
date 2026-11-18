"""
Per camera facts for the delivery metadata (model, lens, field of view,
resolution, frame rate, intrinsic calibration), from the GoPro file and the
OpenICC fisheye calibration (umi.tag_tracking.camera_model.FisheyeCamera,
Kannala-Brandt: r = f * (theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9)).
"""
from typing import Dict, Sequence, Tuple
import json
import numpy as np


def kb_radius(theta: np.ndarray, D: Sequence[float]) -> np.ndarray:
    """
    Kannala-Brandt distorted radius on the normalized image plane at incidence angle theta (rad).
    """
    k = np.asarray(D, dtype=float).reshape(-1)
    return theta * (1 + k[0] * theta ** 2 + k[1] * theta ** 4 + k[2] * theta ** 6 + k[3] * theta ** 8)


def fisheye_fov_deg(K: np.ndarray, D: Sequence[float], image_size: Tuple[int, int]) -> Dict[str, float]:
    """
    Horizontal, vertical and diagonal field of view (deg) covered by the image:
    the incidence angles of the image edges / corners about the principal point.
    """
    w, h = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    theta = np.linspace(0.0, np.pi * 0.75, 20001)
    r = kb_radius(theta, D)
    usable = np.flatnonzero(np.diff(r) > 0)
    theta, r = theta[:usable[-1] + 2], r[:usable[-1] + 2]  # monotonic part of the model

    def angle(dx, dy):
        return float(np.interp(np.hypot(dx / fx, dy / fy), r, theta))

    return {
        'horizontal': float(np.degrees(angle(cx, 0) + angle(w - cx, 0))),
        'vertical': float(np.degrees(angle(0, cy) + angle(0, h - cy))),
        'diagonal': float(np.degrees(angle(cx, cy) + angle(w - cx, h - cy))),
    }


def camera_record(role: str, serial: str, exif: Dict, fps: float, camera) -> Dict[str, str]:
    """
    MCAP metadata record (string values) of one camera. exif: ExifTool metadata of its video
    (GoPro keys); camera: FisheyeCamera of the tag tracking config.
    """
    k = np.asarray(camera.D).reshape(-1)
    K = camera.K
    fov = fisheye_fov_deg(K, k, camera.image_size)
    calibration = json.load(open(camera.intrinsics_json, 'r'))
    return {
        'role': role,
        'camera_serial': serial,
        'model': f"GoPro {exif.get('GoPro:Model', exif.get('QuickTime:Model', ''))}".strip(),
        'firmware': str(exif.get('GoPro:FirmwareVersion', '')),
        'lens': f"{exif.get('GoPro:FieldOfView', '')} (fisheye)".strip(),
        'lens_serial': str(exif.get('UserData:LensSerialNumber', '')),
        'stabilization': str(exif.get('GoPro:ElectronicImageStabilization', '')),
        'field_of_view_deg': f"horizontal {fov['horizontal']:.1f}, vertical {fov['vertical']:.1f}, diagonal {fov['diagonal']:.1f}",
        'resolution': f"{camera.image_size[0]}x{camera.image_size[1]}",
        'frame_rate_hz': f"{fps:.5f}",
        'intrinsics_model': 'kannala_brandt (OpenCV fisheye): r = f * (theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9)',
        'intrinsics_fx_fy_cx_cy_px': ','.join(f'{x:.4f}' for x in (K[0, 0], K[1, 1], K[0, 2], K[1, 2])),
        'intrinsics_k1_k2_k3_k4': ','.join(f'{x:.8f}' for x in k),
        'intrinsics_source': f"OpenImuCameraCalibrator, {calibration.get('nr_calib_images')} board views, "
            f"reprojection error {calibration.get('final_reproj_error', float('nan')):.3f} px",
    }
