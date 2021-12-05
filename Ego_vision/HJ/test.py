import os
import json
import torch
import PIL.Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

data_path = './test/'
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_transform = transforms.Compose([
    transforms.Resize(256),transforms.ToTensor()])

model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=157)
model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('latest.pt', map_location=device), strict=False)
model.eval()

predictions = []
with torch.no_grad():
    for num in sorted(os.listdir(data_path)):
        with open(os.path.join(data_path, f'{num}/{num}.json'), 'r') as js:
            temp = json.load(js)
            imgs = []
            for info in temp['annotations']:
                img_id = info['image_id']
                img_dir = os.path.join(data_path, f'{num}/{img_id}.png')
                img = PIL.Image.open(img_dir).convert('RGB')
                img = test_transform(img)
                imgs.append(img)
            imgs = torch.stack(imgs).cuda()
            prediction = torch.nn.Softmax(dim=1)(model(imgs))
            prediction = torch.mean(prediction, dim=0)
            
            if torch.sum(prediction) != 1: print(torch.sum(prediction))
            predictions.append(prediction.cpu().numpy())

submission = pd.read_csv('sample_submission.csv')
submission.iloc[:,1:] = predictions
submission.to_csv('submission.csv', index=False)