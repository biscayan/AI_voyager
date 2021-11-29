from albumentations.augmentations.geometric.resize import Resize
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import os
import numpy as np
import albumentations
import albumentations.pytorch as albu_torch

class ImageDataset(Dataset):
    def __init__(self, human_file_path1, mask_file_path1, human_file_path2=None, mask_file_path2=None, training=True):
        self.human_file_list = sorted(glob(human_file_path1))
        self.mask_file_list = sorted(glob(mask_file_path1))
        if human_file_path2 and mask_file_path2:
            self.human_file_list = self.human_file_list + sorted(glob(human_file_path2))
            self.mask_file_list = self.mask_file_list + sorted(glob(mask_file_path2))
            
        if training:
            print("Trainig Dataset is loading.")
            self.augmentation = strong_aug()
        else:
            print("Test Dataset is loading.")
            self.augmentation = only_resize()
        
        assert len(self.human_file_list) == len(self.mask_file_list), 'Check File Path. The Number Of Masked Images and Human Images is Different.'
        
    def __len__(self):
        return len(self.human_file_list)
    
    def __getitem__(self, index):
        human_img = np.array(Image.open(self.human_file_list[index]))
        mask_img = np.array(Image.open(self.mask_file_list[index]))
        if len(human_img.shape) == 2:
            human_img = np.repeat(human_img[...,None], 3, axis=-1)
        data = {'image': human_img, 'mask': mask_img}
        aug_img = self.augmentation(**data)
        human_aug, mask_aug = aug_img['image'], aug_img['mask']
        return human_aug, mask_aug
    
class ImageDatasetForRawOnly(Dataset):
    def __init__(self, human_file_path1):
        self.human_file_list = sorted(glob(human_file_path1))
        self.augmentation = only_resize()
        
    def __len__(self):
        return len(self.human_file_list)
    
    def __getitem__(self, index):
        file_name = self.human_file_list[index]
        file_name = os.path.basename(file_name).split('.')[0] + '.png'
        human_img = np.array(Image.open(self.human_file_list[index]))
        height, width = human_img.shape[0], human_img.shape[1]
        if len(human_img.shape) == 2:
            human_img = np.repeat(human_img[...,None], 3, axis=-1)
        data = {'image': human_img}
        aug_img = self.augmentation(**data)
        human_aug = aug_img['image']
        return human_aug, height, width, file_name


def strong_aug(noise_not_weather=0.9, flare_p=0.1, down_p=0.5, force=1, one_of_p=0.5, noise_p=0.98):
    return albumentations.Compose([
        albumentations.Resize(512, 512, always_apply=True),
        albumentations.Normalize(always_apply=True)
        # albu_torch.transforms.ToTensor()
    ])
 
def only_resize(p=0.5):
    return albumentations.Compose([
        albumentations.Resize(512, 512, always_apply=True),
        albumentations.Normalize(always_apply=True)
        # albu_torch.transforms.ToTensor()
    ], p=p)