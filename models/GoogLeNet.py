import torch.nn.functional as F
from torch import nn 
import torch

class Inseption_Block(nn.Module):
    def __init__(self,in_ch, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inseption_Block,self).__init__()
        
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=ch1x1,kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=ch3x3red,kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch3x3red,out_channels=ch3x3,kernel_size=3,padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=ch5x5red,kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch5x5red,out_channels=ch5x5,kernel_size=3,padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1,ceil_mode=True),
            nn.Conv2d(in_channels=in_ch,out_channels=pool_proj,kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class googlenet(nn.Module):
    def __init__(self,num_classes):
        super(googlenet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64,64,kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,192,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.max_pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception1 = Inseption_Block(192,64,96,128,16,32,32)
        self.inception2 = Inseption_Block(256,128,128,192,32,96,64)
        self.max_pool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3 = Inseption_Block(480,192,96,208,16,48,64)
        self.inception4 = Inseption_Block(512,160,112,224,24,64,64)
        self.inception5 = Inseption_Block(512,128,128,256,24,64,64)
        self.inception6 = Inseption_Block(512,112,144,288,32,64,64)
        self.inception7 = Inseption_Block(528,256,160,320,32,128,128)
        self.max_pool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception8 = Inseption_Block(832,256,160,320,32,128,128)
        self.inception9 = Inseption_Block(832,384,192,384,48,128,128)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout2d(0.4)
        self.out = nn.Linear(1024,num_classes)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool2(x)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.max_pool3(x)

        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.max_pool4(x)

        x = self.inception8(x)
        x = self.inception9(x)

        x = self.avg_pool(x)
        x = x.view(-1,1024)
        x = self.dropout(x)
        x = self.out(x)
        return x









        
        
