import os
import glob
import torch
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import Dataset

def run_training():
  image_files = glob.glob(os.path.join(config.DATA_DIR,"*.png"))