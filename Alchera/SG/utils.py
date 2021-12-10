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


def strong_aug(noise_not_weather=0.9, flare_p=0.1, down_p=0.8, force=1, one_of_p=0.5, noise_p=0.9):
    return albumentations.Compose([
        albumentations.augmentations.geometric.transforms.Perspective(),
        albumentations.Resize(256, 256, always_apply=True),
        albumentations.OneOrOther(
            first = albumentations.OneOrOther(
                first = albumentations.Compose([
                    albumentations.OneOf([
                        albumentations.augmentations.transforms.ChannelDropout(p=force),
                        albumentations.augmentations.transforms.ChannelShuffle(p=force),
                        albumentations.augmentations.transforms.ColorJitter(p=force),
                        albumentations.augmentations.transforms.HueSaturationValue(p=force),
                        albumentations.augmentations.transforms.Posterize(p=force),
                        albumentations.augmentations.transforms.CLAHE(p=force),
                        albumentations.augmentations.transforms.RandomBrightnessContrast(p=force),
                        albumentations.augmentations.transforms.RandomGamma(p=force),
                        albumentations.augmentations.transforms.RGBShift(p=force),
                        albumentations.augmentations.transforms.Sharpen(p=force),
                        albumentations.augmentations.transforms.RandomToneCurve(p=force),
                        albumentations.augmentations.transforms.InvertImg(p=force),
                        albumentations.augmentations.transforms.Equalize(mode='pil', p=force),
                        albumentations.augmentations.transforms.Solarize(p=force),
                    ], p=one_of_p),
                    
                    albumentations.OneOf([
                        albumentations.augmentations.transforms.GaussianBlur(p=force),
                        #albumentations.augmentations.transforms.Blur(p=force),
                        albumentations.augmentations.transforms.GaussNoise(p=force),
                        #albumentations.augmentations.transforms.GlassBlur(p=force),
                        #albumentations.augmentations.transforms.MedianBlur(p=force),
                        #albumentations.augmentations.transforms.MotionBlur(p=force),
                        albumentations.augmentations.transforms.MultiplicativeNoise(p=force),
                        albumentations.augmentations.transforms.ISONoise(p=force),
                    ], p=one_of_p),
                    
                    albumentations.OneOf([
                        albumentations.augmentations.transforms.GridDistortion(p=force),
                        albumentations.augmentations.transforms.OpticalDistortion(p=force),
                    ], p=one_of_p),
                    
                    albumentations.OneOf([
                        albumentations.augmentations.transforms.CoarseDropout(p=force),
                        albumentations.augmentations.transforms.MaskDropout(p=force),
                    ], p=one_of_p),
                    ]),
                second = 
                    albumentations.OneOf([
                        albumentations.augmentations.transforms.RandomFog(p=force),
                        albumentations.augmentations.transforms.RandomRain(p=force), #블러와 함께 사용 불가능
                        albumentations.augmentations.transforms.RandomShadow(p=force),
                        albumentations.augmentations.transforms.RandomSnow(p=force),
                        #albumentations.augmentations.transforms.RandomSunFlare(p=flare_p), #주의
                    ], p=force),
                p=noise_not_weather),
            second=albumentations.Compose([]),
        p=noise_p),
        albumentations.augmentations.transforms.Downscale(scale_min=0.01, scale_max=0.10, p=down_p),
        #albumentations.augmentations.transforms.Flip(p=0.5),
        albumentations.Rotate(limit=180, p=force),
        albumentations.Normalize(always_apply=True),
        albu_torch.transforms.ToTensorV2()
    ])
 
def only_resize(p=0.5):
    return albumentations.Compose([
        albumentations.Resize(256, 256, always_apply=True),
        albumentations.Normalize(always_apply=True),
        albu_torch.transforms.ToTensorV2()
    ], p=p)
    
def only_rotate(p=0.5):
    return albumentations.Compose([
        albumentations.augmentations.geometric.transforms.Perspective(),
        albumentations.Resize(256, 256, always_apply=True),
        albumentations.OneOf([
                        albumentations.augmentations.transforms.CoarseDropout(p=1),
                        albumentations.augmentations.transforms.MaskDropout(p=1),
                    ], p=0.5),
        albumentations.Rotate(limit=180, p=1),
        albumentations.Normalize(always_apply=True),
        albu_torch.transforms.ToTensorV2()
    ], p=p)