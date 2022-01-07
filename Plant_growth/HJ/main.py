import os
import torch
import wandb
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GrowthDataset
from model import CompareNet
from train import train

# device setting
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyper parameters
learning_rate = 0.0001
epochs = 100
batch_size = 32
num_workers = 16

# data
bc_comb_df = pd.read_pickle('bc_comb_df') #3161 6812
lt_comb_df = pd.read_pickle('lt_comb_df') #3765 7801
bc_train = bc_comb_df.iloc[:2500]
bc_valid = bc_comb_df.iloc[2500:]
lt_train = lt_comb_df.iloc[:3000]
lt_valid = lt_comb_df.iloc[3000:]

train_set = pd.concat([bc_train, lt_train])
valid_set = pd.concat([bc_valid, lt_valid])

train_dataset = GrowthDataset(train_set, 'train')
valid_dataset = GrowthDataset(valid_set, 'valid')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# model
model = CompareNet().to(device)
model = torch.nn.DataParallel(model).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(train_loader, valid_loader, model, optimizer, device, epochs, batch_size)