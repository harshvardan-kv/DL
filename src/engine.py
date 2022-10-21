import config
import torch
import tqdm

def train_fn(model,data_loader, optimizer):
  model.train()
  fin_loss = 0
  tk = tqdm(data_loader,total = len(data_loader))
  for data in tk:
    for k,v in data.items():
      data[k] = v.to(config.DEVICE)
    optimizer.zero_grad()
    _,loss = model(**data)
    optimizer.step()
    fin_loss += loss.item()
  return fin_loss/len(data_loader)

def eval_fn(model,data_loader, optimizer):
  model.eval()
  fin_loss = 0
  fin_pred = []
  tk = tqdm(data_loader,total = len(data_loader))
  for data in tk:
    for k,v in data.items():
      data[k] = v.to(config.DEVICE)
    batch_pred,loss = model(**data)

    fin_loss += loss.item()
    fin_pred.append(batch_pred)
  return fin_loss/len(data_loader)
