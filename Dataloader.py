from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image


class MyDataSet(Dataset):
    def __init__(self,images,labels,transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        Pil_image = Image.fromarray(img)
        if self.transforms is not None:
            img = self.transforms(Pil_image)
        #img = np.moveaxis(img,-1,0)
        #img = img / img.max()
        #img = img.transpose(2,0,1)
        lbl = self.labels[idx]
        return img, torch.tensor(lbl)
        #return torch.tensor(img),torch.tensor(lbl)