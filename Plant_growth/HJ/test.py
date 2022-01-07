import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import GrowthDataset
from model import CompareNet

os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_set = pd.read_csv('test_dataset/test_data.csv')
test_set['l_root'] = test_set['before_file_path'].map(lambda x: './test_dataset/' + x.split('_')[1] + '/' + x.split('_')[2])
test_set['r_root'] = test_set['after_file_path'].map(lambda x: './test_dataset/' + x.split('_')[1] + '/' + x.split('_')[2])
test_set['l_path'] = test_set['l_root'] + '/' + test_set['before_file_path'] + '.png'
test_set['r_path'] = test_set['r_root'] + '/' + test_set['after_file_path'] + '.png'
test_set.drop(['before_file_path', 'after_file_path', 'l_root', 'r_root'], axis=1, inplace=True)
test_set.rename(columns = {'l_path' : 'before_file_path', 'r_path' : 'after_file_path'}, inplace = True)

test_dataset = GrowthDataset(test_set, 'test')
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16)

model = CompareNet().to(device)
model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('latest.pt', map_location=device))
model.eval()

test_value = []
with torch.no_grad():
    for test_before, test_after in tqdm(test_data_loader):
        test_before = test_before.to(device)
        test_after = test_after.to(device)
        logit = model(test_before, test_after)
        value = logit.squeeze(1).detach().cpu().float()
        
        test_value.extend(value)
        
submission = pd.read_csv('sample_submission.csv')
_sub = torch.FloatTensor(test_value)

__sub = _sub.numpy()
__sub[np.where(__sub<1)] = 1

submission['time_delta'] = __sub
submission.to_csv('result.csv', index=False)