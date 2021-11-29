import torch, os
from torch.utils.data import DataLoader
from utils import ImageDataset
import torch.nn as nn
import torch.nn.functional as F
import torch 
import wandb
import logging
from torch import optim
from torchmetrics import IoU
from dice_score import dice_loss
from tqdm import tqdm
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3,4,5,6'

GPU_NUM = 0
 # 원하는 GPU 번호 입력
DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE) # change allocation of current GPU

EPOCHS = 3000
BATCH_SIZE = 12
LEARNING_RATE = 0.001
EPSILON = 1e-7
NUM_WORKERS = 8
PREFETCH_FACTOR = 48

STACK_NUM = 10
CHANNUL_NUM_LIST = [3, 32, 64, 128, 256]
LABEL_NUM = 15
amp = False

TRAIN_HUMAN_PATH = '/home/mts/taegyu/nunbody/train_dataset/Train/Image/*' 
TRAIN_MASK_PATH = '/home/mts/taegyu/nunbody/train_dataset/Train/Mask/*' 
VAL_HUMAN_PATH = None 
VAL_MASK_PATH = None 
TEST_HUMAN_PATH = '/home/mts/taegyu/Inbody_Segmentation_testset_Participant_pixelchange_gray-smooth/image/*' 
TEST_MASK_PATH = '/home/mts/taegyu/Inbody_Segmentation_testset_Participant_pixelchange_gray-smooth/mask/*' 

MODEL_LEARNING_HISTORY_PATH = 'log_dir'
MODEL_SAVE_PATH = 'model_save'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# logging
experiment = wandb.init(project="my-test-project", entity="cau")
experiment.config.update(dict(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE))
                         
logging.info(f'''Starting training:
        Epochs:          {EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
    ''')    


model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="noisy-student",     # use `imagenet` pre-trained weights for encoder initialization
    activation='softmax',                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=15,                      # model output channels (number of classes in your dataset)
).to(DEVICE)

SAVED_MODEL = '/home/mts/taegyu/inbody_segmentation/model_save/latest.pt'

# 모델을 불러와서 재학습할 시 model dict 불러오기
# checkpoint = torch.load(SAVED_MODEL, map_location=DEVICE)
# for key in list(checkpoint.keys()):
#     if 'module.' in key:
#         checkpoint[key.replace('module.', '')] = checkpoint[key]
#         del checkpoint[key]
        
# model.load_state_dict(checkpoint)


model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


train_dataset = ImageDataset(TRAIN_HUMAN_PATH, TRAIN_MASK_PATH, human_file_path2=VAL_HUMAN_PATH, mask_file_path2=VAL_MASK_PATH)
val_dataset = ImageDataset(TEST_HUMAN_PATH, TEST_MASK_PATH, training=False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
iou = IoU(num_classes=15, dist_sync_on_step=True).to(DEVICE)

step = 0
min_loss = float('inf')
for epoch in tqdm(range(1, EPOCHS)):
    model.train()
    epoch_loss = 0
    for batch, data in tqdm(enumerate(train_dataloader)):
        humans, labels = data
        humans = humans.to(DEVICE).float()
        labels = labels.to(DEVICE).long()
        humans = humans.permute(0, 3, 1, 2)
        masks_pred = model(humans)
        loss = 0
        iou_temp = 0
        loss = dice_loss(masks_pred, F.one_hot(labels, 15).permute(0, 3, 1, 2).float(), multiclass=True)
        total_iou = iou(masks_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        epoch_loss += loss.item()
        experiment.log({
                    'train loss': loss.item(),
                    'step': step,
                    'epoch': epoch,
                    'total_iou' : total_iou
                })
        
        print('train loss:', loss.item())
    
        
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_iou2 = 0
        for batch, data in enumerate(val_dataloader):
            humans, labels = data
            humans = humans.to(DEVICE).float()
            labels = labels.to(DEVICE).long()
            humans = humans.permute(0, 3, 1, 2)
            masks_pred = model(humans)
            loss = dice_loss(masks_pred, F.one_hot(labels, 15).permute(0, 3, 1, 2).float(), multiclass=True)
            total_loss += loss.item()
            total_iou2 += iou(masks_pred, labels)
        print('test :', total_loss/len(val_dataloader))
        experiment.log({
                'test loss': total_loss/len(val_dataloader),
                'test_iou' : total_iou2 / len(val_dataloader)
                }) 
    if min_loss > total_loss:
        min_loss = total_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH + '/min_loss_b4_noaug.pt')
    torch.save(model.state_dict(), MODEL_SAVE_PATH + '/latest_b4_noaug.pt')