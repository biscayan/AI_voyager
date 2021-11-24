import torch, os
from torch.utils.data import DataLoader
from utils import ImageDatasetForRawOnly
from PIL import Image
import numpy as np
import argparse
from torchvision.transforms.functional import resize
import cv2

parser = argparse.ArgumentParser(description='segmentation test')
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()


GPU_NUM = 6 # 원하는 GPU 번호 입력
DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 1
STACK_NUM = 10
CHANNUL_NUM_LIST = [3, 32, 64, 128, 256]
LABEL_NUM = 15

SAVED_MODEL = './Unet++_b4_new.pt' # 모델 파일 위치를 적어주세요
TEST_HUMAN_PATH = '{}/*'.format(args.data_path) # inference 할 이미지 경로
NEW_MASK_PATH = '{}'.format(args.save_path) # inference 후 저장할 경로

model = torch.load(SAVED_MODEL)
model.eval()

val_dataset = ImageDatasetForRawOnly(TEST_HUMAN_PATH)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,  pin_memory=True)


with torch.no_grad():
    for batch, data in enumerate(val_dataloader):
        humans, height, width, file_name  = data
        with torch.no_grad():
            human = humans.to(DEVICE)
            human = human.permute(0, 3, 1, 2)
            output = model(human)
            output = torch.argmax(output, dim=1)[0] 
            output = resize(output.to("cpu").unsqueeze(0), size=[height , width]);
            output = output[0]
            cv2.imwrite( os.path.join(NEW_MASK_PATH, file_name[0].split(sep='.')[0]) +'.png', output.to('cpu').detach().numpy())
            
            
  
