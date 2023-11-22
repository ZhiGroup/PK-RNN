from torch.optim import Adadelta
from model import PK_RNN
from dataset import Example, collate_batch
from torch.utils.data import DataLoader
import torch

model = PK_RNN()
model_save_path = './best_model_wts.pt'
trainloader = DataLoader(Example(), batch_size=50, collate_fn=collate_batch)  # replace this with your own dataset
validloader = DataLoader(Example(), batch_size=50, collate_fn=collate_batch)  # replace this with your own dataset
optimizer = Adadelta(model.parameters(),
                     eps=0.0001,
                     lr=0.0501133197,
                     weight_decay=0.2)
n_epoch = 1500
patience = 20
best_valid_loss = torch.inf
bad_cnt = 0
for i in range(n_epoch):
    for batch in trainloader:
        optimizer.zero_grad()
        _, _, _, loss, _ = model(batch)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        loss = 0
        cnt = 0
        for batch in validloader:
            batch_size = batch[0].shape[0]
            mse_loss = model(batch)[-1]
            loss += loss * batch_size
            cnt += batch_size
        valid_loss = loss/cnt
        print(f"epoch {i + 1}: validation loss {valid_loss}")
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), model_save_path)
            best_valid_loss = valid_loss
            bad_cnt = 0
        else:
            bad_cnt += 1
        if bad_cnt == patience:
            break


