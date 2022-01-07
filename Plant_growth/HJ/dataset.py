import albumentations
import albumentations.pytorch as albu_torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class GrowthDataset(Dataset):
    def __init__(self, combination_df, type=None):
        self.combination_df = combination_df
        self.type = type
        self.train_transform = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
            albumentations.OneOf([
                albumentations.augmentations.transforms.ChannelDropout(p=1),
                albumentations.augmentations.transforms.ChannelShuffle(p=1),
                albumentations.augmentations.transforms.CLAHE(p=1),
                albumentations.augmentations.transforms.RandomGamma(p=1),
                albumentations.augmentations.transforms.RGBShift(p=1),
            ], p=0.5),
            albumentations.OneOf([
                albumentations.augmentations.transforms.Flip(p=1),
                albumentations.augmentations.transforms.VerticalFlip(p=1),
                albumentations.augmentations.transforms.HorizontalFlip(p=1),
            ], p=0.5),                
            albumentations.OneOf([
                albumentations.augmentations.transforms.GaussNoise(p=1),
                albumentations.augmentations.transforms.MultiplicativeNoise(p=1),
                albumentations.augmentations.transforms.ISONoise(p=1),
            ], p=0.5),
            albumentations.OneOf([
                albumentations.augmentations.transforms.GridDistortion(p=1),
                albumentations.augmentations.transforms.OpticalDistortion(p=1),
            ], p=0.5),
            albumentations.Normalize(always_apply=True),
            albu_torch.transforms.ToTensorV2()])
        
        self.valid_transform = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
            albumentations.Normalize(always_apply=True),
            albu_torch.transforms.ToTensorV2()])

    def __getitem__(self, idx):
        before_image = Image.open(self.combination_df.iloc[idx]['before_file_path'])
        after_image = Image.open(self.combination_df.iloc[idx]['after_file_path'])        
        before_image = np.array(before_image)
        after_image = np.array(after_image)

        if self.type == 'test':
            before_image = self.valid_transform(image=before_image)
            after_image = self.valid_transform(image=after_image)
            before_image = before_image['image']
            after_image = after_image['image']
            return before_image, after_image

        time_delta = self.combination_df.iloc[idx]['time_delta']
        
        if self.type == 'train':        
            before_image = self.train_transform(image=before_image)
            after_image = self.train_transform(image=after_image)
            before_image = before_image['image']
            after_image = after_image['image']
        elif self.type == 'valid':
            before_image = self.valid_transform(image=before_image)
            after_image = self.valid_transform(image=after_image)
            before_image = before_image['image']
            after_image = after_image['image']

        return before_image, after_image, time_delta

    def __len__(self):
        return len(self.combination_df)