import torch
import os
import argparse
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from PIL import Image
from utils import ImageDatasetForRawOnly

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="불러올 이미지 폴더 경로")
parser.add_argument("--save_path", help="저장할 이미지 폴더 경로")
option = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
LABEL_NUM = 14

SAVED_MODEL = 'model_save/latest.pt'
TEST_HUMAN_PATH = option.data_path
NEW_MASK_PATH = option.save_path
os.makedirs(NEW_MASK_PATH, exist_ok=True)

model = smp.UnetPlusPlus('efficientnet-b5', encoder_depth=3, encoder_weights='imagenet', decoder_channels=(256, 128, 64), in_channels=3, classes=LABEL_NUM, activation="tanh")
model = torch.nn.DataParallel(model).to(DEVICE)
model.load_state_dict(torch.load(SAVED_MODEL, map_location=DEVICE), strict=False)
model.eval()

test_dataset = ImageDatasetForRawOnly(TEST_HUMAN_PATH)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

with torch.no_grad():
    for batch, data in enumerate(test_dataloader):
        humans, height, width, file_name = data
        humans = humans.to(DEVICE).float()
        preds = model(humans)
        mask_image = torch.argmax(preds, dim=1)
        mask_image = mask_image.squeeze()
        mask_image = mask_image.cpu().numpy()
        mask_image = Image.fromarray(mask_image.astype(np.uint8))
        mask_image = mask_image.resize((int(width[0]), int(height[0])))
        mask_image.save(NEW_MASK_PATH + '/' + file_name[0])