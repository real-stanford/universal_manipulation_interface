import cv2

img_path = "/home/sungjoon/Downloads/right_Infrared.png"
img = cv2.imread(img_path)

# mask_path = "/home/sungjoon/Desktop/study/universal_manipulation_interface/helper/im_l_infrared_mask.png"
# mask_path = "/home/sungjoon/Desktop/study/universal_manipulation_interface/helper/im_l_Infrared_mask.png"
mask_path = "/home/sungjoon/Desktop/study/universal_manipulation_interface/helper/im_r_Infrared_mask.png"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img.shape[:2] != mask.shape:
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

img[mask > 0] = 0

cv2.imwrite("./asdf.png", img)