import math
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=0)

def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)

class reConv(nn.mdule):
    def __init__(self):
        channels = [3,256,256,256,256,256]
        self.iterTimes = 10
        self.rawConv = conv1x1(channels[0],channels[1])
        self.Conv1 = conv3x3(channels[1],channels[2])
        self.Conv2 = conv3x3(channels[2],channels[3])
        self.Conv3 = conv1x1(channels[3],channels[4])
        self.outConv = conv1x1(channels[4]*self.iterTimes,self.channels[5])

    def forward(self,x):
        iterTimes = 10
        x = self.rawConv(x)
        for i in range(iterTimes):
            x = self.Conv1(x)
            x = self.Conv2(x)
            x = self.Conv3(x)
            if i == 0:
                xfusion = x
            xfusion = torch.cat(xfusion,x)
        out = self.outConv(xfusion)
            