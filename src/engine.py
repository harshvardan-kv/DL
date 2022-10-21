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
    
    _,loss = model(**data)