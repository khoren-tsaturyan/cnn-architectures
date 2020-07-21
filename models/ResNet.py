from torch import nn
import torch.nn.functional as F
import torch

class Residual_Block(nn.Module):
    expansion = 1
    def __init__(self,in_ch,out_ch,stride=1):
        super(Residual_Block,self).__init__()
        self.main_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(num_features=out_ch),
        )
        self.shortcut = lambda x:x
        if stride>1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(num_features=out_ch))

    def forward(self,x):
        s = self.shortcut(x)
        x = self.main_layers(x)
        return F.relu(torch.add(x,s))


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_ch,out_ch,stride=1):
        super(Bottleneck,self).__init__()
        self.main_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch*4,kernel_size=1,bias=False),
            nn.BatchNorm2d(num_features=out_ch*4)
        )

        self.shortcut = lambda x:x
        if stride>1 or in_ch != out_ch*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_ch,out_channels=out_ch*4,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(num_features=out_ch*4))

    def forward(self,x):
        s = self.shortcut(x)
        x = self.main_layers(x)
        return F.relu(torch.add(x,s))


class ResNet(nn.Module):
    def __init__(self,Block,n_layers,n_classes,n_start=16):
        super(ResNet,self).__init__()
        self.Block = Block
        self.n_layers = n_layers
        self.n_classes = n_classes        
        self.layers = self._make_layers()

    def forward(self,x):
        x = self.layers(x)
        return x

    def _make_layers(self):
        layers = [
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        ]
        prev_channel = 64 
        expansion = 1
        channels = []
        for i,n in enumerate(self.n_layers):
            channels+=n*[64*(2**i)]

        for channel in channels:
            stride = 1 if  channel*expansion==prev_channel else 2
            layers+= [self.Block(prev_channel,channel,stride)]
            expansion = self.Block.expansion
            prev_channel = channel*self.Block.expansion

        layers+= [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512*self.Block.expansion,self.n_classes)
        ]

        return nn.Sequential(*layers)


def resnet18(n_classes):
    model = ResNet(Residual_Block, [2, 2, 2, 2],n_classes)
    return model


def resnet34(n_classes):
    model = ResNet(Residual_Block, [3, 4, 6, 3],n_classes)
    return model
    

def resnet50(n_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3],n_classes)
    return model


def resnet101(n_classes):
    model = ResNet(Bottleneck, [3, 4, 23, 3],n_classes)
    return model

    
def resnet152(n_classes):
    model = ResNet(Bottleneck, [3, 8, 36, 3],n_classes)
    return model
