"""
Per camera facts for the delivery metadata (model, lens, field of view,
resolution, frame rate, intrinsic calibration), from the GoPro file and the
OpenICC fisheye calibration (umi.tag_tracking.camera_model.FisheyeCamera,
Kannala-Brandt: r = f * (theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9)).
"""
from typing import Dict, Sequence, Tuple
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
    record = {
        'role': role,
        'serial': serial,
        'model': f"GoPro {exif.get('GoPro:Model', exif.get('QuickTime:Model', ''))}".strip(),
        'firmware': str(exif.get('GoPro:FirmwareVersion', '')),
        'lens': str(exif.get('GoPro:FieldOfView', '')),
        'stabilization': 'off',
        'fov_horizontal_deg': f"{fov['horizontal']:.1f}",
        'fov_vertical_deg': f"{fov['vertical']:.1f}",
        'fov_diagonal_deg': f"{fov['diagonal']:.1f}",
        'resolution': f"{camera.image_size[0]}x{camera.image_size[1]}",
        'frame_rate_hz': f"{fps:.5f}",
        'intrinsics_model': 'kannala_brandt',
        'fx': f'{K[0, 0]:.4f}', 'fy': f'{K[1, 1]:.4f}', 'cx': f'{K[0, 2]:.4f}', 'cy': f'{K[1, 2]:.4f}',
        'k1': f'{k[0]:.8f}', 'k2': f'{k[1]:.8f}', 'k3': f'{k[2]:.8f}', 'k4': f'{k[3]:.8f}',
    }
    return {key: value for key, value in record.items() if value not in ('', 'GoPro')}
