import torch
from dataset import *
import os


dataset_STR = STRDataset(root='train_data',labelPath='train_high_crop_list2.txt',charsetPath='chinese_cht_dict.txt')
data_loader = torch.utils.data.DataLoader(
    dataset_STR, batch_size = 1, shuffle = False, num_workers = 0, collate_fn = collate_fn
)
charsDict = dataset_STR.charsDict
lost_word = []
for _,labels in data_loader:
    label = list(labels)[0]
    for char in label:
        try:
            tmp = charsDict[char]
        except KeyError:
            if char not in lost_word:
                lost_word.append(char)

with open('lost_word.txt','w',encoding = 'utf-8') as f:
    lost_word = sorted(lost_word)
    for char in lost_word:
        f.write(char+'\n')