# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import numpy as np
import cv2
from PIL import Image
import yaml

# %%
def pad_image(img, pixels, value=0):
    return cv2.copyMakeBorder(img,
        pixels, pixels, pixels, pixels,
        borderType=cv2.BORDER_CONSTANT,
        value=value)

def generate_marker_imgs(aruco_dict, start_id, n_markers, 
                        marker_size_mm, margin_mm,
                        pad_extra_mm=0,
                        text_size_mm=0,
                        border_pixels=1,
                        dpi=300):
    mm_per_inch = 25.4
    marker_size_pixel = round(marker_size_mm / mm_per_inch * dpi)
    pad_pixel = round(margin_mm / mm_per_inch * dpi)
    assert pad_pixel > 1

    imgs = list()
    for marker_id in range(start_id, start_id+n_markers):
        marker_img = cv2.aruco.generateImageMarker (
            dictionary=aruco_dict, id=marker_id, 
            sidePixels=marker_size_pixel, borderBits=1)
        bpd = border_pixels
        wpd = pad_pixel - bpd
        padded_img = pad_image(marker_img, wpd, 255)
        padded_img = pad_image(padded_img, bpd, 0)

        if text_size_mm > 0:
            dict_size = len(aruco_dict.bytesList)
            n_digits = len(str(dict_size - 1))
            text = str(marker_id)
            zeros = '0' * (n_digits - len(text))
            text = zeros + text
            raw_text_size = cv2.getTextSize(text=(str(dict_size-1)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)
            raw_pix_height = raw_text_size[0][1]
            target_pix_height = text_size_mm / mm_per_inch * dpi
            scale = target_pix_height / raw_pix_height

            cv2.putText(padded_img, text, org=(0,round(target_pix_height*1.2)), 
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=scale, color=0)
            
        if pad_extra_mm > 0:
            padded_img = pad_image(padded_img, 
                round(pad_extra_mm / mm_per_inch * dpi), 255)

        imgs.append(padded_img)
    return imgs


# %%
@click.command()
@click.option('-o', '--output_dir', required=True)
def main(output_dir):
    output_dir = os.path.expanduser(output_dir)

    yaml_name = 'aruco_config.yaml'
    text_scale = 1/3

    # config for gripper markers
    # occupy id 0-11
    n_finger_markers = 2
    n_gripper_cube_markers = 4
    n_grippers = 2

    finger_marker_size_mm = 16
    finger_marker_margin_mm = 2
    finger_marker_pad_mm = 5

    gripper_cube_marker_size_mm = 60
    gripper_cube_marker_margin_mm = 5
    gripper_cube_marker_pad_mm = 3

    cubes_pdf_fname = 'aruco_cubes_letter.pdf'
    # config for table cube markers
    # occupy id 12-19
    n_table_cube_markers = 8

    table_cube_marker_size_mm = 160
    table_cube_marker_margin_mm = 10
    table_cube_marker_pad_mm = 20

    dpi = 300

    # config for ArUco dict
    predefined_dict_name = 'DICT_4X4_50'
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, predefined_dict_name))
    
    # marker size in unit meters
    marker_size_map = {
        'default': table_cube_marker_size_mm / 1000
    }

    # generate gripper pdf (11x17 print)
    gripper_imgs = list()
    for gripper_id in range(n_grippers):
        start_id = gripper_id * (n_finger_markers + n_gripper_cube_markers)
        finger_markers = generate_marker_imgs(
            aruco_dict=aruco_dict,
            start_id=start_id, n_markers=n_finger_markers,
            # adjust size here
            marker_size_mm=finger_marker_size_mm, 
            margin_mm=finger_marker_margin_mm, 
            pad_extra_mm=finger_marker_pad_mm,
            text_size_mm=finger_marker_margin_mm * text_scale,
            dpi=dpi)
        for i in range(start_id, start_id + n_finger_markers):
            marker_size_map[i] = finger_marker_size_mm / 1000

        start_id = start_id + n_finger_markers
        cube_markers = generate_marker_imgs(
            aruco_dict=aruco_dict,
            start_id=start_id, n_markers=n_gripper_cube_markers,
            # adjust size here
            marker_size_mm=gripper_cube_marker_size_mm, 
            margin_mm=gripper_cube_marker_margin_mm, 
            pad_extra_mm=gripper_cube_marker_pad_mm,
            text_size_mm=gripper_cube_marker_margin_mm * text_scale,
            dpi=dpi)
        for i in range(start_id, start_id + n_gripper_cube_markers):
            marker_size_map[i] = gripper_cube_marker_size_mm / 1000
        
        finger_marker_size = finger_markers[0].shape[1]
        cube_marker_size = cube_markers[0].shape[1]
        pad_shape = (finger_marker_size, cube_marker_size - finger_marker_size*2)
        finger_row = list(finger_markers)
        finger_row.insert(1, np.full(pad_shape, fill_value=255, dtype=np.uint8))
        finger_row_img = np.concatenate(finger_row, axis=1)

        left_col = np.concatenate([finger_row_img] + cube_markers[:2], axis=0)
        right_col = np.concatenate([finger_row_img] + cube_markers[2:], axis=0)
        gripper_img = np.concatenate([left_col, right_col], axis=1)

        gripper_imgs.append(gripper_img)

        # save
        im = Image.fromarray(gripper_img)

        fname = f'aruco_gripper_{gripper_id}_letter.pdf'
        fpath = os.path.join(output_dir, fname)
        im.save(fpath, resolution=dpi)

    # generate table marker pdf
    start_id = n_grippers * (n_finger_markers + n_gripper_cube_markers)
    table_markers = generate_marker_imgs(
        aruco_dict=aruco_dict,
        start_id=start_id, n_markers=n_table_cube_markers,
        # adjust size here
        marker_size_mm=table_cube_marker_size_mm, 
        margin_mm=table_cube_marker_margin_mm, 
        pad_extra_mm=table_cube_marker_pad_mm,
        text_size_mm=table_cube_marker_margin_mm * text_scale,
        border_pixels=0,
        dpi=dpi)
    
    for i in range(start_id, start_id + n_table_cube_markers):
        marker_size_map[i] = table_cube_marker_size_mm / 1000
    
    fpath = os.path.join(output_dir, cubes_pdf_fname)
    marker_imgs = [Image.fromarray(x) for x in table_markers]
    marker_imgs[0].save(fpath, 
        resolution=dpi, save_all=True, append_images=marker_imgs[1:])

    aruco_config = {
        'aruco_dict': {
            'predefind': predefined_dict_name
        },
        'marker_size_map': marker_size_map
    }
    fpath = os.path.join(output_dir, yaml_name)
    yaml.dump(aruco_config, stream=open(fpath, 'w'))
    

# %%
if __name__ == "__main__":
    main()
