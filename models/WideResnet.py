from torch import nn
import torch.nn.functional as F
import torch


class Residual_Block(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super(Residual_Block,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,1,1,bias=False)
        self.shortcut = lambda x:x
        if in_ch!=out_ch:
            self.shortcut = nn.Conv2d(in_ch,out_ch,1,stride,bias=False)

    def forward(self,x):
        x = F.relu(self.bn1(x),inplace=True)
        s = self.shortcut(x)
        x = self.conv1(x)
        x = F.relu(self.bn2(x),inplace=True)
        x = self.conv2(x)*0.2
        return x.add_(s) 


class wideresnet(nn.Module):
    def __init__(self,n_classes,n_start=16):
        super(wideresnet,self).__init__()
        self.start = nn.Conv2d(3,n_start,3,1,1,bias=False)
        layers = [self.start]
        n_chanals = [n_start]
        n_groups = 3
        n_blocks = 3 
        for i in range(n_groups):
            n_chanals.append(n_start*(2**i))
            stride = 2 if i>0 else 1
            layers+= self.make_groups(n_blocks,n_chanals[i],n_chanals[i+1],stride)

        layers+=[
                nn.BatchNorm2d(n_chanals[-1]),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_chanals[-1],n_classes)
        ]

        self.features = nn.Sequential(*layers)

    @staticmethod
    def make_groups(n,in_ch,out_ch,stride):
        start = Residual_Block(in_ch,out_ch,stride)
        rest = [Residual_Block(out_ch,out_ch) for _ in range(n-1)]
        return [start]+rest


    def forward(self,x):
        return self.features(x)


