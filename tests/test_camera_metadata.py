"""
umi.tag_tracking.camera_metadata: field of view of a fisheye calibration, camera record.

python tests/test_camera_metadata.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    image_size = (2704, 2028)
    K = np.array([[1200.0, 0, 1352], [0, 1200.0, 1014], [0, 0, 1]])
    D = np.array([0.05, 0.02, 0.006, -0.011])


def test_record():
    exif = {'GoPro:Model': 'HERO12 Black', 'GoPro:FieldOfView': 'Wide', 'GoPro:FirmwareVersion': 'H23.01',
        'UserData:LensSerialNumber': 'L1', 'GoPro:ElectronicImageStabilization': 'N/A'}
    r = camera_record('right_wrist', 'C1', exif, 59.94005994, FakeCamera())
    assert all(isinstance(v, str) for v in r.values())
    assert r['model'] == 'GoPro HERO12 Black' and r['lens'] == 'Wide' and r['stabilization'] == 'off'
    assert r['resolution'] == '2704x2028' and r['frame_rate_hz'] == '59.94006' and r['intrinsics_model'] == 'kannala_brandt'
    assert r['fx'] == '1200.0000' and r['k1'] == '0.05000000' and float(r['fov_horizontal_deg']) > 100
    assert 'lens_serial' not in r and 'intrinsics_source' not in r
    # values the file does not have are left out
    assert 'firmware' not in camera_record('head', 'C2', {'GoPro:Model': 'HERO12 Black'}, 59.94, FakeCamera())


if __name__ == '__main__':
    test_fov()
    test_record()
    print("all passed")
