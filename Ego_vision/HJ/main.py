import os
import torch
import wandb
import torch.optim as optim
import torch.nn as nn
import albumentations
import albumentations.pytorch as albu_torch
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from data_processing import data_split, TrainDataset, ValDataset
from train import train

# device setting
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyper parameters
learning_rate = 0.0001
epochs = 100
batch_size = 32
num_workers = 8
force = 1
one_of_p = 0.5

# data
train_imgs, val_imgs, train_labels, val_labels, label_info = data_split('./train/')

train_transform = albumentations.Compose([
        albumentations.Resize(256, 256, always_apply=True),
        albumentations.OneOrOther(
            first = albumentations.OneOrOther(
                albumentations.OneOf([
                    albumentations.augmentations.transforms.ChannelDropout(p=force),
                    albumentations.augmentations.transforms.ChannelShuffle(p=force),
                    albumentations.augmentations.transforms.CLAHE(p=force),
                    albumentations.augmentations.transforms.RandomGamma(p=force),
                    albumentations.augmentations.transforms.RGBShift(p=force),
                ], p=one_of_p),
                albumentations.OneOf([
                    albumentations.augmentations.transforms.GaussNoise(p=force),
                    albumentations.augmentations.transforms.MultiplicativeNoise(p=force),
                        albumentations.augmentations.transforms.ISONoise(p=force),
                ], p=one_of_p),
                albumentations.OneOf([
                    albumentations.augmentations.transforms.GridDistortion(p=force),
                    albumentations.augmentations.transforms.OpticalDistortion(p=force),
                ], p=one_of_p),
                ),
            second=albumentations.Compose([])
            ),
        albumentations.Normalize(always_apply=True),
        albu_torch.transforms.ToTensorV2()])

val_transform = albumentations.Compose([
        albumentations.Resize(256, 256, always_apply=True),
        albumentations.Normalize(always_apply=True),
        albu_torch.transforms.ToTensorV2()])
        
train_dataset = TrainDataset(train_imgs, train_labels, label_info, transform = train_transform)
val_dataset = ValDataset(val_imgs, val_labels, label_info, transform = val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# model
model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=len(train_dataset.label_info))
model = torch.nn.DataParallel(model).to(device)
summary(model, input_size=(3,256,256))
        
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# experiment
wandb.init(project="Ego-vision", entity="biscayan")
wandb.config = {
"learning_rate": learning_rate,
"epochs": epochs,
"batch_size": batch_size
}
wandb.watch(model, log='all')

train(train_loader, val_loader, model, optimizer, loss_fn, device, epochs)