import albumentations
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Classification:
  def __init__(self, image_path , target=None , resize =None):
    self.image_path = image_path
    self.target = target
    self.resize = resize
    self.aug = albumentations.Compose(
      [
        albumentations.Normalize(always_apply=True)
      ]
    )

  def __len__(self):
    return len(self.image_path)

  def __getitem__(self,item):
    image = Image.open(self.image_path[item]).convert("RGB")
    targets = self.target[item]
    if self.resize is not None:
      image = image.resize((self.resize[1],self.resize[0]),resample = Image.BILINEAR)

 

    image = np.array(image)
    
    augmented = self.aug(image = image)
    image = augmented["image"]
    image = np.transpose(image,(2,0,1)).astype(np.float32)

    return {
      "images": torch.tensor(image,dtype=torch.float),
      "target": torch.tensor(targets,dtype=torch.float)
    }


