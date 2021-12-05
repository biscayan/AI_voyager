import os
import json
import PIL
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def data_split(data_path):
    imgs, labels = [], []
    for num in sorted(os.listdir(data_path)):
        with open(os.path.join(data_path, f'{num}/{num}.json'), 'r') as js:
            temp = json.load(js)
            for info in temp['annotations']:
                img_id = info['image_id']
                imgs.append(os.path.join(data_path, f'{num}/{img_id}.png'))
                labels.append(temp['action'][0])

    label_info = {label:i for i, label in enumerate(sorted(set(labels)))}
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(imgs, labels, test_size=0.2, random_state=34, stratify=labels)
    
    return train_imgs, val_imgs, train_labels, val_labels, label_info

class TrainDataset(Dataset):
    def __init__(self, train_imgs, train_labels, label_info, transform=None):
        self.imgs = train_imgs
        self.labels = train_labels
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, idx):
        img = PIL.Image.open(self.imgs[idx]).convert('RGB')
        img = np.array(img)
        if self.transform: 
            img = self.transform(image=img)
            img = img['image']

        label = self.label_info[self.labels[idx]]
        return img, label

    def __len__(self):
        return len(self.imgs)
    
class ValDataset(Dataset):
    def __init__(self, val_imgs, val_labels, label_info, transform=None):
        self.imgs = val_imgs
        self.labels = val_labels
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, idx):
        img = PIL.Image.open(self.imgs[idx]).convert('RGB')
        img = np.array(img)
        if self.transform: 
            img = self.transform(image=img)
            img = img['image']

        label = self.label_info[self.labels[idx]]
        return img, label

    def __len__(self):
        return len(self.imgs)