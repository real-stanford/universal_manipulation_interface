from typing import Dict, Tuple

import json
import pathlib
import math
import copy
import numpy as np
import cv2
import scipy.interpolate as si


RGB_IMG_SHAPE = (1080, 1920)
IR_IMG_SHAPE = (480, 640)


def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.227, z_tolerance=0.0015):
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None

    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width


def parse_realsense_intrinsics(json_data: dict) -> Dict[str, np.ndarray]:
    assert json_data['intrinsic_type'] == 'PINHOLE'
    intr_data = json_data['intrinsics']
    
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # pinhole parameters
    fx = intr_data['fx']
    fy = intr_data['fy']

    px = intr_data['principal_pt_x']
    py = intr_data['principal_pt_y']
    
    # Kannala-Brandt non-linear parameters for distortion
    # kb8 = [
    #     intr_data['radial_distortion_1'],
    #     intr_data['radial_distortion_2'],
    #     intr_data['radial_distortion_3'],
    #     intr_data['radial_distortion_4']
    # ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [fx, 0, px],
            [0, fy, py],
            [0, 0, 1]
        ], dtype=np.float64),
        # 'D': np.array([kb8]).T
        'D': np.zeros((1,5))
    }
    return opencv_intr_dict


def convert_intrinsics_resolution(
        opencv_intr_dict: Dict[str, np.ndarray], 
        target_resolution: Tuple[int, int]
        ) -> Dict[str, np.ndarray]:
    """
    Convert fisheye intrinsics parameter to a different resolution,
    assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.
    """
    iw, ih = opencv_intr_dict['DIM']
    iK = opencv_intr_dict['K']
    ifx = iK[0,0]
    ify = iK[1,1]
    ipx = iK[0,2]
    ipy = iK[1,2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([
        [ofx, 0, opx],
        [0, ofy, opy],
        [0, 0, 1]
    ], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict['DIM'] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict['K'] = oK
    return out_intr_dict


def get_aruco_dict(predefined:str
                   ) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, predefined))

# ================= ArUcO tag =====================
def parse_aruco_config(aruco_config_dict: dict):
    """
    example:
    aruco_dict:
        predefined: DICT_4X4_50
    marker_size_map: # all unit in meters
        default: 0.15
        12: 0.2
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict['aruco_dict'])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict['marker_size_map']
    default_size = marker_size_map.get('default', None)
    
    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        out_marker_size_map[marker_id] = size
    
    result = {
        'aruco_dict': aruco_dict,
        'marker_size_map': out_marker_size_map
    }
    return result

def detect_localize_aruco_tags(
        img: np.ndarray, 
        aruco_dict: cv2.aruco.Dictionary, 
        marker_size_map: Dict[int, float], 
        realsense_intr_dict: Dict[str, np.ndarray], 
        refine_subpix: bool=True):
    K = realsense_intr_dict['K']
    D = realsense_intr_dict['D']

    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        image=img, dictionary=aruco_dict, parameters=param)
    if len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue
        
        marker_size_m = marker_size_map[this_id]
        
        # undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
            this_corners, marker_size_m, K, D)
        
        tag_dict[this_id] = {
            'rvec': rvec.squeeze(),
            'tvec': tvec.squeeze(),
            'corners': this_corners.squeeze()
        }
    return tag_dict

# draw

def canonical_to_pixel_coords(coords, img_shape):
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts

def pixel_coords_to_canonical(pts, img_shape):
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords

def get_rgb_gripper_canonical_polygon():
    left_pts = (
        (210, 1080),
        (360, 865),
        (1550, 825),
        (1730, 1080),
    )
    
    left_coords = pixel_coords_to_canonical(left_pts, RGB_IMG_SHAPE)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    # coords = np.stack([left_coords, right_coords])
    coords = np.stack([left_coords])
    return coords


def get_rgb_finger_canonical_polygon():
    left_pts = (
        (430, 865),
        (510, 700),
        (790, 580),
        (1130, 580),
        (1380, 700),
        (1490, 825),
    )
    
    left_coords = pixel_coords_to_canonical(left_pts, RGB_IMG_SHAPE)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    # coords = np.stack([left_coords, right_coords])
    coords = np.stack([left_coords])
    return coords

def draw_rgb_predefined_mask(img, color=(0,0,0),*,mirror=True, gripper=True, finger=True, use_aa=False):
    all_coords = list()

    if mirror:
        ...
    if gripper:
        all_coords.extend(get_rgb_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_rgb_finger_canonical_polygon())

    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img


def draw_im_l_infrared_mask(img, color=(0,0,0),*,mirror=True, gripper=True, finger=True, use_aa=False):
    all_coords = list()

    pts = (
        (80, 480),
        (80, 350),
        (164, 280),
        (248, 240),
        (340, 240),
        (420, 280),
        (470, 346),
        (470, 480),
    )

    coords = np.stack([pixel_coords_to_canonical(pts, IR_IMG_SHAPE)])
    all_coords.extend(coords)

    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img

def draw_im_r_infrared_mask(img, color=(0,0,0),*,mirror=True, gripper=True, finger=True, use_aa=False):
    all_coords = list()

    pts = (
        (0, 480),
        (0, 330),
        (90, 275),
        (190, 240),
        (280, 240),
        (335, 275),
        (360, 350),
        (360, 480)
    )

    coords = np.stack([pixel_coords_to_canonical(pts, IR_IMG_SHAPE)])
    all_coords.extend(coords)

    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img