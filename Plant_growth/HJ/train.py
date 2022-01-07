import torch
import wandb

def train(train_loader, valid_loader, model, optimizer, device, epochs, batch_size):
    for epoch in range(epochs):
        train_losses = []
        model.train()
        for step, (before_image, after_image, time_delta) in enumerate(train_loader):
            before_image = before_image.to(device)
            after_image = after_image.to(device)
            time_delta = time_delta.to(device)

            logit = model(before_image, after_image)
            train_loss = (torch.sum(torch.abs(logit.squeeze(1).float() - time_delta.float())) /
                        torch.LongTensor([batch_size]).squeeze(0).to(device))
            train_losses.append(train_loss.detach().cpu())
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        valid_losses = []
        model.eval()
        with torch.no_grad():
            for valid_before, valid_after, time_delta in valid_loader:
                valid_before = valid_before.to(device)
                valid_after = valid_after.to(device)
                valid_time_delta = time_delta.to(device)

                logit = model(valid_before, valid_after)
                valid_loss = (torch.sum(torch.abs(logit.squeeze(1).float() - valid_time_delta.float())) /
                            torch.LongTensor([batch_size]).squeeze(0).to(device))
                valid_losses.append(valid_loss.detach().cpu())

        print(f'Epoch : {epoch} | Train loss : {sum(train_losses)/len(train_losses):.3f} | Valid loss : {sum(valid_losses)/len(valid_losses):.3f}')
        torch.save(model.state_dict(), 'latest.pt')