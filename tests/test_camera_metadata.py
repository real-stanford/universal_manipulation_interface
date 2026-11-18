"""
umi.tag_tracking.camera_metadata: field of view of a fisheye calibration, camera record.

python tests/test_camera_metadata.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import numpy as np
from umi.tag_tracking.camera_metadata import fisheye_fov_deg, camera_record


def test_fov():
    # equidistant (k = 0): r = f theta, so the half widths in pixels / f are the angles
    K = np.array([[1000.0, 0, 1000], [0, 1000, 500], [0, 0, 1]])
    fov = fisheye_fov_deg(K, [0, 0, 0, 0], (2000, 1000))
    assert abs(fov['horizontal'] - np.degrees(2.0)) < 0.01 and abs(fov['vertical'] - np.degrees(1.0)) < 0.01
    assert abs(fov['diagonal'] - np.degrees(2 * np.hypot(1.0, 0.5))) < 0.01
    # positive k1 (more distortion): the same pixels cover a smaller angle
    assert fisheye_fov_deg(K, [0.05, 0, 0, 0], (2000, 1000))['horizontal'] < fov['horizontal']


class FakeCamera:
    def __init__(self, path):
        self.intrinsics_json = path
        self.image_size = (2704, 2028)
        self.K = np.array([[1200.0, 0, 1352], [0, 1200.0, 1014], [0, 0, 1]])
        self.D = np.array([0.05, 0.02, 0.006, -0.011])


def test_record():
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump({'nr_calib_images': 25, 'final_reproj_error': 0.425}, f)
    try:
        exif = {'GoPro:Model': 'HERO12 Black', 'GoPro:FieldOfView': 'Wide', 'GoPro:FirmwareVersion': 'H23.01',
            'UserData:LensSerialNumber': 'L1', 'GoPro:ElectronicImageStabilization': 'N/A'}
        r = camera_record('right_wrist', 'C1', exif, 59.94005994, FakeCamera(f.name))
    finally:
        os.unlink(f.name)
    assert all(isinstance(v, str) for v in r.values())
    assert r['model'] == 'GoPro HERO12 Black' and r['lens'] == 'Wide (fisheye)' and r['resolution'] == '2704x2028'
    assert r['frame_rate_hz'] == '59.94006' and r['intrinsics_fx_fy_cx_cy_px'].startswith('1200.0000,1200.0000,1352')
    assert '25 board views' in r['intrinsics_source'] and r['field_of_view_deg'].startswith('horizontal ')


if __name__ == '__main__':
    test_fov()
    test_record()
    print("all passed")
