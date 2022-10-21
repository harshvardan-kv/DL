import os
import glob
from scipy.sparse.compressed import operator
import torch
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import Dataset
from model import Captcha
import engine
def run_training():
  image_files = glob.glob(os.path.join(config.DATA_DIR,"*.png"))
  target_orig = [x.split("/")[-1][:-4] for x in image_files]
  targets = [[c for c in x] for x in target_orig]
  targets_flat = [items for item in targets for items in item]

  lbl_enc = preprocessing.LabelEncoder()
  lbl_enc.fit(targets_flat)

  target_enc = [lbl_enc.transform(x) for x in targets]
  target_enc = np.array(target_enc)+1
  print(target_enc)
  print(len(lbl_enc.classes_))

  train_imgs,test_imgs, train_targets, test_targets, train_orig_targets , test_orig_targets  = model_selection.train_test_split(image_files,target_enc,target_orig,test_size = 0.1, random_state=42)

  print(train_imgs[:2],train_targets[:2],train_orig_targets[:2])

  train_Dataset = Dataset.Classification(train_imgs,train_targets,resize=(config.IMAGE_HEIGHT,config.IMAGE_WIFTH))

  test_Dataset = Dataset.Classification(test_imgs,test_targetsresize=(config.IMAGE_HEIGHT,config.IMAGE_WIFTH))

  train_DataLoader = torch.utils.data.DataLoader(
    train_Dataset,
    batch_size = config.BATCH_SIZE,
    num_workers = config.NUM_WORKERS,
    shuffle = True
  )

  test_DataLoader = torch.utils.data.DataLoader(
    test_Dataset,
    batch_size = config.BATCH_SIZE,
    num_workers = config.NUM_WORKERS,
    shuffle = False
  )

  model = Captcha(num_chars=len(lbl_enc.classes_))
  model.to(config.DEVICE)

  optimizer = torch.optim.Adam(model.parameters(),lr = 3e-4)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.8, patience=5, verbose=True
  )

  for epoch in range(config.EPOCH):
    train_loss= engine.train_fn(model,train_DataLoader,optimizer)
    train_loss= engine.train_fn(model,train_DataLoader,optimizer)
    valid_preds,valid_loss = engine.eval_fn(model,train_DataLoader)
if __name__ == "__main__":
  run_training()


