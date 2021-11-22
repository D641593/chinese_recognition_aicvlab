import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import re

img_except_h, img_except_w = 32,512
class STRDataset(torch.utils.data.Dataset):
    def __init__(self, root, labelPath):
        self.root = root
        self.labelPath = labelPath
        self.imgPaths = []
        self.labels = []
        with open(os.path.join(self.root, self.labelPath),'r',encoding = 'utf-8') as f:
            txt = f.readlines()
            for row in txt:
                row = row.split('\t')
                path = row[0]
                self.imgPaths.append(os.path.join(self.root,path))
                self.labels.append(random.randint(0,4))
        self.imgPaths, self.labels = (list(i) for i in zip(*sorted(zip(self.imgPaths,self.labels))))

    def __getitem__(self, idx):
        img = Image.open(self.imgPaths[idx])
        label = self.labels[idx]
        img = img.rotate(90*label,expand = True)
        trans = get_transforms()
        img = trans(img)
        return img,label

    def __len__(self):
        return len(self.imgPaths)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transforms(h,w):
    ratio = img_except_h / h
    ratio_w = int(ratio * w)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_except_h,ratio_w))
    ])

