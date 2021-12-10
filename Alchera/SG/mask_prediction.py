import torch, os
from torch.utils.data import DataLoader
from model_small import StackedHourGlass
from utils_small_image import ImageDatasetForRawOnly
from PIL import Image
import numpy as np

DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1

STACK_NUM = 13
CHANNUL_NUM_LIST = [3, 32, 64, 128, 256]
LABEL_NUM = 14


SAVED_MODEL = 'iou_metrics_model/latest.pt'

TEST_HUMAN_PATH = 'test_dataset/human/*'
TEST_HUMAN_PATH = 'train_dataset/train_total_human/*'
NEW_MASK_PATH = 'inferenced_mask'
os.makedirs(NEW_MASK_PATH, exist_ok=True)
    
model = StackedHourGlass(stack_num=STACK_NUM, channel_num_list=CHANNUL_NUM_LIST, label_num=LABEL_NUM).to(DEVICE)
checkpoint = torch.load(SAVED_MODEL, map_location=DEVICE)['model']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
model.load_state_dict(checkpoint)
model.eval()

val_dataset = ImageDatasetForRawOnly(TEST_HUMAN_PATH)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, prefetch_factor=4, pin_memory=False)

with torch.no_grad():
    for batch, data in enumerate(val_dataloader):
        humans, height, width, file_name = data
        humans = humans.to(DEVICE).float()
        logit_list = model(humans)
        pred = logit_list[-1]
        mask_image = torch.argmax(pred, dim=1)
        mask_image = mask_image.squeeze()
        mask_image = mask_image.numpy() * 19
        mask_image = Image.fromarray(mask_image.astype(np.uint8))
        mask_image = mask_image.resize((int(width[0]), int(height[0])))
        mask_image.save(NEW_MASK_PATH + '/' + file_name[0])
        