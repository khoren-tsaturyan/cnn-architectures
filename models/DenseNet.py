import torch 
from torch import nn 
from torch.nn import functional as F 


class TransitionLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(TransitionLayer,self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        self.avg_pool = nn.AvgPool2d(2)
    
    def forward(self,x):
        x = F.relu(self.bn(x))
        x = self.conv(x)
        x = self.avg_pool(x)
        return x  


class DenseBlock(nn.Module):
    def __init__(self,in_ch,growth_rate):
        super(DenseBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch,4*growth_rate,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3, padding=1,bias=False)

    def forward(self,x):
        y = F.relu(self.bn1(x))
        y = self.conv1(y)
        y = F.relu(self.bn2(y))
        y = self.conv2(y)
        x = torch.cat([y,x],1)        
        return x
        

class DenseNet(nn.Module):
    def __init__(self,n_blocks,growth_rate,n_classes):
        super(DenseNet,self).__init__()
        self.growth_rate = growth_rate
        in_channels = 2*growth_rate 
        reduction = 0.5

        self.conv1 = nn.Conv2d(3,in_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.max_pool = nn.MaxPool2d(3,stride=2,padding=1)

        self.dense_block1 = self._make_dense_layers(in_channels,n_blocks[0])
        in_channels+= growth_rate*n_blocks[0]
        out_channels = int(in_channels*reduction)
        self.trans1 = TransitionLayer(in_channels,out_channels)
        in_channels = out_channels

        self.dense_block2 = self._make_dense_layers(in_channels,n_blocks[1])
        in_channels+= growth_rate*n_blocks[1]
        out_channels = int(in_channels*reduction)
        self.trans2 = TransitionLayer(in_channels,out_channels)
        in_channels = out_channels

        self.dense_block3 = self._make_dense_layers(in_channels,n_blocks[2])
        in_channels+= growth_rate*n_blocks[2]
        out_channels = int(in_channels*reduction)
        self.trans3 = TransitionLayer(in_channels,out_channels)
        in_channels = out_channels

        self.dense_block4 = self._make_dense_layers(in_channels,n_blocks[3])
        in_channels+= growth_rate*n_blocks[3]

        self.bn = nn.BatchNorm2d(in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels,n_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.dense_block1(x)
        x = self.trans1(x)
        x = self.dense_block2(x)
        x = self.trans2(x)
        x = self.dense_block3(x)
        x = self.trans3(x)
        x = self.dense_block4(x)
        x = F.relu(self.bn(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def _make_dense_layers(self,in_ch,n_block):
        layers = []
        for _ in range(n_block):
            layers.append(DenseBlock(in_ch,self.growth_rate))
            in_ch+= self.growth_rate
        return nn.Sequential(*layers)


def densenet121(n_classes):
    return DenseNet([6,12,24,16],32,n_classes=n_classes) 

def densenet169(n_classes):
    return DenseNet([6,12,32,32],32,n_classes) 

def densenet201(n_classes):
    return DenseNet([6,12,48,32],32,n_classes) 

def densenet264(n_classes):
    return DenseNet([6,12,64,48],32,n_classes) 

