import os
import cv2
import numpy as np
from tqdm import tqdm

def process_image(img_path, output_path, size=299):
    img = cv2.imread(img_path)
    if img is None: return
    
    # 1. CLAHE Enhancement (The Pro Trick)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # 2. Square Padding (Don't stretch the coin!)
    h, w = img.shape[:2]
    sh, sw = size, size
    interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC
    aspect = w/h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_top, pad_bot, pad_left, pad_right = 0,0,0,0

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, 
                                    borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    
    cv2.imwrite(output_path, scaled_img)

def run_pipeline(min_images=10):
    raw_path = 'data/raw/CN_dataset_v1/dataset_types'
    proc_path = 'data/processed'
    
    os.makedirs(proc_path, exist_ok=True)
    
    for coin_type in tqdm(os.listdir(raw_path)):
        folder = os.path.join(raw_path, coin_type)
        images = os.listdir(folder)
        
        if len(images) >= min_images:
            target_dir = os.path.join(proc_path, coin_type)
            os.makedirs(target_dir, exist_ok=True)
            
            for img_name in images:
                process_image(os.path.join(folder, img_name), 
                              os.path.join(target_dir, img_name))

if __name__ == "__main__":
    run_pipeline()