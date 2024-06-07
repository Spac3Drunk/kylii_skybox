import os
import cv2
from PIL import Image
import numpy as np
import random
import math
import string

def saveImageList(imageList):
    # Generate a random name with 5 random letters
    randName = ''.join(random.choice(string.ascii_letters) for i in range(5))
    
    # Iterate over the images and save them
    for i, img in enumerate(imageList):
        path = os.path.join('results', f'{randName}-{i}.jpg')
        print(path)
        cv2.imwrite(path, img)

def get_left_half(image):
    cropped_img = image[:, :image.shape[1]//2]
    return cropped_img

def get_right_half(image):
    cropped_img = image[:, image.shape[1]//2:]
    return cropped_img

def split_for_ratio_top(image, ratio_h, ratio_w):
    ratio = ratio_h/ratio_w
    img_ratio = image.shape[0]/image.shape[1]
    if ratio > img_ratio:
        raise ValueError('inputed ratio height is superior compared to the image ratio height')
    else:
        final_height = math.floor((ratio/img_ratio)*image.shape[0])
        top_part = image[:final_height, :]
        bottom_part = image[final_height:, :]
        return [top_part, bottom_part]

def vShapeThis(image):
    # Create a mask for the diagonal line starting from the top right corner
    mask_right = np.tri(*image.shape[:2], k=0, dtype=bool)
    # Create a mask for the diagonal line starting from the top left corner
    mask_left = np.tri(*image.shape[:2], k=-1, dtype=bool)
    # Apply both masks to the image
    image[mask_right] = 0
    image[mask_left] = 0
    return image.astype(np.uint8)