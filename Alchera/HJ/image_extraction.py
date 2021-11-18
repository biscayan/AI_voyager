import os
import shutil
import tqdm
import numpy as np
from glob import glob
from PIL import Image

def make_human_img(mask_file_path, raw_file_path, new_file_path):
    os.makedirs(new_file_path, exist_ok=True)
    
    mask_files = glob(mask_file_path+'/*')
    raw_files = glob(raw_file_path+'/*')
    
    mask_files = set([os.path.basename(one_file).split('.')[0] for one_file in mask_files])
    
    for raw_file in tqdm.tqdm(raw_files):
        if os.path.basename(raw_file).split('.')[0] in mask_files:
            shutil.copyfile(raw_file, new_file_path + '/' + os.path.basename(raw_file))
            
def mask_head_0(mask_file_path, new_file_path):
    os.makedirs(new_file_path, exist_ok=True)
    
    mask_files = glob(mask_file_path+'/*')
    
    for mask_file in tqdm.tqdm(mask_files):
        temp_img = Image.open(mask_file)
        np_img = np.array(temp_img)
        np_img[np_img==14] = 0
        pil_image = Image.fromarray(np_img)
        pil_image.save(new_file_path + '/' + os.path.basename(mask_file))

# COCO dataset을 처리하려면 밑의 주석을 풀면 됩니다.
# if __name__ == '__main__':
#     BASE_PATH = 'train_dataset'
#     train_mask_path = 'train_mask'
#     train_raw_path = 'train2014'
#     train_human_path = 'train_human'
#     train_mask_no_head_path = 'train_mask_no_head'
#     val_mask_path = 'val_mask'
#     val_raw_path = 'val2014'
#     val_human_path = 'val_human'
#     val_mask_no_head_path = 'val_mask_no_head'
#     make_human_img(BASE_PATH + '/' + train_mask_path, BASE_PATH + '/' + train_raw_path, BASE_PATH + '/' + train_human_path)
#     make_human_img(BASE_PATH + '/' + val_mask_path, BASE_PATH + '/' + val_raw_path, BASE_PATH + '/' + val_human_path)
#     mask_head_0(BASE_PATH + '/' + train_mask_path, BASE_PATH + '/' + train_mask_no_head_path)
#     mask_head_0(BASE_PATH + '/' + val_mask_path, BASE_PATH + '/' + val_mask_no_head_path)

# 눈바디 데이터셋을 처리하려면 밑의 주석을 풀면 됩니다.
if __name__ == '__main__':
    BASE_PATH = 'test_dataset'
    MASK_PATH = 'mask'
    HUMAN_PATH = 'human'
    human_files = glob(BASE_PATH + '/*.jpg')
    mask_files = glob(BASE_PATH + '/*.png')
    os.makedirs(BASE_PATH + '/' + HUMAN_PATH, exist_ok=True)
    os.makedirs(BASE_PATH + '/' + MASK_PATH, exist_ok=True)
    for human_file in tqdm.tqdm(human_files):
        shutil.copyfile(human_file, BASE_PATH + '/' + HUMAN_PATH + '/' + os.path.basename(human_file))
        
    for mask_file in tqdm.tqdm(mask_files):
        temp_img = Image.open(mask_file)
        np_img = np.array(temp_img)
        np_img[np_img==14] = 0
        pil_image = Image.fromarray(np_img)
        pil_image.save(BASE_PATH + '/' + MASK_PATH + '/' + os.path.basename(mask_file))