import torch
import wandb

def calc(output, target):
    predict = torch.argmax(output, 1)
    correct = (predict == target).sum().item()
    return correct

def train(train_loader, val_loader, model, optimizer, loss_fn, device, epochs):
    
    min_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, train_data in enumerate(train_loader):
            train_x, train_y = train_data
            train_x, train_y = train_x.to(device), train_y.to(device)

            output = model(train_x)
            trn_loss = loss_fn(output, train_y)
            train_loss += trn_loss
            train_acc = 100 * calc(output, train_y) / train_y.size(0)

            optimizer.zero_grad()
            trn_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for j, val_data in enumerate(val_loader):
                val_x, val_y = val_data
                val_x, val_y = val_x.to(device), val_y.to(device)
                
                output = model(val_x)
                val_loss = loss_fn(output, val_y)
                valid_loss += val_loss
                valid_acc = 100 * calc(output, val_y) / val_y.size(0)
                    
        wandb.log({"Train_loss":train_loss/len(train_loader), "Valid_loss":valid_loss/len(val_loader), 
           "Train_acc":train_acc, "Valid_acc":valid_acc})
        
        print(f'Epoch : {epoch} | Train loss : {train_loss/len(train_loader):.3f} | Valid loss : {valid_loss/len(val_loader):.3f}')
        print(f'Epoch : {epoch} | Train acc : {train_acc:.3f} | Valid acc : {valid_acc:.3f}')
        
        if min_loss > valid_loss:
            min_loss = valid_loss
            torch.save(model.module.state_dict(), 'min_loss.pt')
        torch.save(model.module.state_dict(), 'latest.pt')