import torch
import os
import wandb
import segmentation_models_pytorch as smp
from torchmetrics import IoU
from torch.utils.data import DataLoader
from torchsummary import summary
from utils import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_WORKERS = 4
LABEL_NUM = 14

TRAIN_HUMAN_PATH = 'train_dataset/train_human/*'
TRAIN_MASK_PATH = 'train_dataset/train_mask_no_head/*' 
VAL_HUMAN_PATH = 'train_dataset/val_human/*' 
VAL_MASK_PATH = 'train_dataset/val_mask_no_head/*' 
TEST_HUMAN_PATH = 'test_dataset/human/*'
TEST_MASK_PATH = 'test_dataset/mask/*' 

MODEL_SAVE_PATH = 'model_save'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

model = smp.UnetPlusPlus('efficientnet-b5', encoder_depth=3, encoder_weights='imagenet', decoder_channels=(256, 128, 64), in_channels=3, classes=LABEL_NUM, activation="tanh")
model = torch.nn.DataParallel(model).to(DEVICE)
summary(model, input_size=(3,256,256))

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
iou = IoU(num_classes=LABEL_NUM, dist_sync_on_step=True).to(DEVICE)

train_dataset = ImageDataset(TRAIN_HUMAN_PATH, TRAIN_MASK_PATH, human_file_path2=VAL_HUMAN_PATH, mask_file_path2=VAL_MASK_PATH)
val_dataset = ImageDataset(TEST_HUMAN_PATH, TEST_MASK_PATH, training=False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

step = 0
min_loss = float('inf')

wandb.init(project="Alchera", entity="biscayan")
wandb.config = {
  "learning_rate": LEARNING_RATE,
  "epochs": EPOCHS,
  "batch_size": BATCH_SIZE
}
wandb.watch(model, log='all')

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss, train_iou = 0, 0
    for batch, data in enumerate(train_dataloader):
        humans, labels = data
        humans = humans.to(DEVICE).float()
        labels = labels.to(DEVICE).long()
        
        preds = model(humans)
        trn_loss = loss_fn(preds, labels)
        trn_iou = iou(preds, labels)
        train_loss += trn_loss
        train_iou += trn_iou
        
        optimizer.zero_grad()
        trn_loss.backward()
        optimizer.step()
        step += 1
        
    model.eval()
    with torch.no_grad():
        valid_loss, valid_iou = 0, 0
        for batch, data in enumerate(val_dataloader):
            humans, labels = data
            humans = humans.to(DEVICE).float()
            labels = labels.to(DEVICE).long()
            
            preds = model(humans)
            val_loss = loss_fn(preds, labels)
            val_iou = iou(preds, labels)
            valid_loss += val_loss
            valid_iou += val_iou

    wandb.log({"Train_loss":train_loss/len(train_dataloader), "Valid_loss":valid_loss/len(val_dataloader), 
               "Train_IoU":train_iou/len(train_dataloader), "Valid_IoU":valid_iou/len(val_dataloader)})
    print(f'Epoch : {epoch} | Train loss : {train_loss/len(train_dataloader):.3f} | Valid loss : {valid_loss/len(val_dataloader):.3f}')
    print(f'Epoch : {epoch} | Train IoU : {train_iou/len(train_dataloader):.3f} | Valid IoU : {valid_iou/len(val_dataloader):.3f}')
    
    if min_loss > valid_loss:
        min_loss = valid_loss
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH + '/min_loss.pt')
    torch.save(model.module.state_dict(), MODEL_SAVE_PATH + '/latest.pt')