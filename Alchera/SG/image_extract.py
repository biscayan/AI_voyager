from glob import glob
from PIL import Image
import os, shutil, tqdm
import numpy as np

# find human images which match mask images from datasets
def make_human_img(mask_file_path, raw_file_path, new_file_path):
    os.makedirs(new_file_path, exist_ok=True)
    
    mask_files = glob(mask_file_path+'/*')
    raw_files = glob(raw_file_path+'/*')
    
    mask_files = [os.path.basename(one_file).split('.')[0] for one_file in mask_files]
    
    for raw_file in tqdm.tqdm(raw_files):
        if os.path.basename(raw_file).split('.')[0] in mask_files:
            shutil.copyfile(raw_file, new_file_path + '/' + os.path.basename(raw_file))
            
# erase head part
def mask_head_0(mask_file_path, new_file_path):
    os.makedirs(new_file_path, exist_ok=True)
    
    mask_files = glob(mask_file_path+'/*')
    
    for mask_file in tqdm.tqdm(mask_files):
        temp_img = Image.open(mask_file)
        np_img = np.array(temp_img)
        np_img[np_img==14] = 0
        pil_image = Image.fromarray(np_img)
        pil_image.save(new_file_path + '/' + os.path.basename(mask_file))

# copy images to other folders
def copy_img_folder_to_folder(raw_file_path, new_file_path):
    os.makedirs(new_file_path, exist_ok=True)
    
    raw_files = glob(raw_file_path+'/*')
    for raw_file in tqdm.tqdm(raw_files):
        shutil.copyfile(raw_file, new_file_path + '/' + os.path.basename(raw_file))
        
# vertical flip images
def make_filp_img(raw_file_path, new_file_path):
    os.makedirs(new_file_path, exist_ok=True)
    
    raw_files = glob(raw_file_path+'/*')
    for raw_file in tqdm.tqdm(raw_files):
        temp_img = Image.open(raw_file)
        flip_img = temp_img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_img.save(new_file_path + '/' + os.path.basename(raw_file))

# vertical flip mask. body segmentations are different whether they are on left or right, so it is needed to change directions.
def make_flip_mask(mask_file_path, new_file_path):
    os.makedirs(new_file_path, exist_ok=True)
    
    mask_files = glob(mask_file_path+'/*')
    
    for mask_file in tqdm.tqdm(mask_files):
        temp_img = Image.open(mask_file)
        # 2=3, 4=5, 6=7, 8=9, 10=11, 12=13
        np_img = np.array(temp_img)
        np_img[np_img==2] = 100
        np_img[np_img==3] = 2
        np_img[np_img==100] = 3
        
        np_img[np_img==4] = 100
        np_img[np_img==5] = 4
        np_img[np_img==100] = 5
        
        
        np_img[np_img==6] = 100
        np_img[np_img==7] = 6
        np_img[np_img==100] = 7
        
        np_img[np_img==8] = 100
        np_img[np_img==9] = 8
        np_img[np_img==100] = 9
        
        np_img[np_img==10] = 100
        np_img[np_img==11] = 10
        np_img[np_img==100] = 11
        
        np_img[np_img==12] = 100
        np_img[np_img==13] = 12
        np_img[np_img==100] = 13
        pil_image = Image.fromarray(np_img)
        pil_image.save(new_file_path + '/' + os.path.basename(mask_file))

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

# change root for testdatasets for Dataset.
# if __name__ == '__main__':
#     BASE_PATH = 'test_dataset'
#     MASK_PATH = 'mask'
#     HUMAN_PATH = 'human'
#     human_files = glob(BASE_PATH + '/*.jpg')
#     mask_files = glob(BASE_PATH + '/*.png')
#     os.makedirs(BASE_PATH + '/' + HUMAN_PATH, exist_ok=True)
#     os.makedirs(BASE_PATH + '/' + MASK_PATH, exist_ok=True)
#     for human_file in tqdm.tqdm(human_files):
#         shutil.copyfile(human_file, BASE_PATH + '/' + HUMAN_PATH + '/' + os.path.basename(human_file))
        
#     for mask_file in tqdm.tqdm(mask_files):
#         temp_img = Image.open(mask_file)
#         np_img = np.array(temp_img)
#         np_img[np_img==14] = 0
#         pil_image = Image.fromarray(np_img)
#         pil_image.save(BASE_PATH + '/' + MASK_PATH + '/' + os.path.basename(mask_file))

# put all coco dataset in one file.
# if __name__ == '__main__':
#     BASE_PATH = 'train_dataset'
#     train_human_path = 'train_human'
#     train_mask_no_head_path = 'train_mask_no_head'
#     val_human_path = 'val_human'
#     val_mask_no_head_path = 'val_mask_no_head'
    
#     new_human_path = 'train_total_human'
#     new_mask_path = 'train_total_mask'
    
#     copy_img_folder_to_folder(BASE_PATH + '/' + train_human_path, BASE_PATH + '/' + new_human_path)
#     copy_img_folder_to_folder(BASE_PATH + '/' + val_human_path, BASE_PATH + '/' + new_human_path)
#     copy_img_folder_to_folder(BASE_PATH + '/' + train_mask_no_head_path, BASE_PATH + '/' + new_mask_path)
#     copy_img_folder_to_folder(BASE_PATH + '/' + val_mask_no_head_path, BASE_PATH + '/' + new_mask_path)
        
    
# if __name__ == '__main__':
#     BASE_PATH = 'train_dataset'
#     new_human_path = 'train_total_human'
#     new_mask_path = 'train_total_mask'
#     flip_human_path = 'train_total_flip_human'
#     flip_mask_path = 'train_total_flip_mask'
    
#     make_filp_img(BASE_PATH + '/' + new_human_path, BASE_PATH + '/' + flip_human_path)
#     make_filp_img(BASE_PATH + '/' + new_mask_path, BASE_PATH + '/' + flip_mask_path)

# if __name__ == '__main__':
#     BASE_PATH = 'train_dataset'
#     flip_mask_path = 'train_total_flip_mask'
#     flip_mask_path_test = 'train_total_flip_mask_check'
#     make_flip_mask(BASE_PATH + '/' + flip_mask_path, BASE_PATH + '/' + flip_mask_path_test)

# if __name__ == '__main__':
#     BASE_PATH = 'train_dataset'
#     flip_mask_path = 'train_total_flip_mask'
#     flip_mask_path_test = 'train_total_flip_mask_check'
#     check_target = 'COCO_train2014_000000000322.png'
#     temp_img = Image.open(BASE_PATH + '/' + flip_mask_path + '/' + check_target)
#     np_img = np.array(temp_img)
#     np_img = np_img * 10
#     pil_image = Image.fromarray(np_img)
#     pil_image.save('check1.png')
#     temp_img = Image.open(BASE_PATH + '/' + flip_mask_path_test + '/' + check_target)
#     np_img = np.array(temp_img)
#     np_img = np_img * 10
#     pil_image = Image.fromarray(np_img)
#     pil_image.save('check2.png')
    