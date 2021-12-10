import torch, os
from torch.utils.data import DataLoader
from model import StackedHourGlass
from utils import ImageDataset
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from torchmetrics import IoU

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10000
BATCH_SIZE = 35
LEARNING_RATE = 0.00001
EPSILON = 1e-7

STACK_NUM = 13
CHANNUL_NUM_LIST = [3, 32, 64, 128, 256]
LABEL_NUM = 14

TRAIN_HUMAN_PATH = 'train_dataset/train_total_human/*'
TRAIN_MASK_PATH = 'train_dataset/train_total_mask/*'
VAL_HUMAN_PATH = 'train_dataset/train_total_flip_human/*'
VAL_MASK_PATH = 'train_dataset/train_total_flip_mask/*'
TEST_HUMAN_PATH = 'test_dataset/human/*'
TEST_MASK_PATH = 'test_dataset/mask/*'

SAVE_PAHT = 'iou_metric_model/'
os.makedirs(SAVE_PAHT, exist_ok=True)
    
model = StackedHourGlass(stack_num=STACK_NUM, channel_num_list=CHANNUL_NUM_LIST, label_num=LABEL_NUM)
model = torch.nn.DataParallel(model).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, min_lr=0.00001, verbose=True, factor=0.5)

# checkpoint = torch.load('iou_metrics_not_noise_L13/latest.pt', map_location=DEVICE)
# for key in list(checkpoint.keys()):
#     if 'module.' in key:
#         checkpoint[key.replace('module.', '')] = checkpoint[key]
#         del checkpoint[key]
#model.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = smp.losses.DiceLoss(mode='multiclass')
iou = IoU(num_classes=14, dist_sync_on_step=True).to(DEVICE)

train_dataset = ImageDataset(TRAIN_HUMAN_PATH, TRAIN_MASK_PATH, human_file_path2=VAL_HUMAN_PATH, mask_file_path2=VAL_MASK_PATH)
val_dataset = ImageDataset(TEST_HUMAN_PATH, TEST_MASK_PATH, training=False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3, prefetch_factor=24, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3, prefetch_factor=24)
writer = SummaryWriter('log_dir')
step = 0
min_loss = float('inf')
max_iou = -float('inf')
for epoch in range(1, EPOCHS):
    model.train()
    for batch, data in enumerate(train_dataloader):
        humans, labels = data
            
        humans = humans.to(DEVICE).float()
        labels = labels.to(DEVICE).long()
        
        logit_list = model(humans)
        loss = 0
        for logit in logit_list[-1:]:
            loss += loss_fn(logit, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        writer.add_scalar("Loss/train_step", loss, step)
        
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_iou = 0
        for batch, data in enumerate(val_dataloader):
            humans, labels = data
            
            humans = humans.to(DEVICE).float()
            labels = labels.to(DEVICE).long()
            
            logit_list = model(humans)
            loss = 0
            iou_temp = 0
            for logit in logit_list[-1:]:
                loss += (loss_fn(logit, labels) * len(labels))
                iou_temp += (iou(logit, labels) * len(labels))
                
            total_loss += loss
            total_iou += iou_temp
        writer.add_scalar("Loss/val_step", total_loss/len(val_dataset), epoch)
        writer.add_scalar("IoU/val_step", total_iou/len(val_dataset), epoch)
        
    scheduler.step(total_loss)

    if min_loss > total_loss:
        min_loss = total_loss
        torch.save(model.state_dict(), SAVE_PAHT + 'min_loss.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, SAVE_PAHT + 'min_loss_optimizer.pt')
    if max_iou < total_iou:
        max_iou = total_iou
        torch.save(model.state_dict(), SAVE_PAHT + 'max_iou.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, SAVE_PAHT + 'latest.pt')