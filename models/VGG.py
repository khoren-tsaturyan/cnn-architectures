import torch.nn.functional as F
from torch import nn


cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self,vgg_type,num_classes):
        super(VGG,self).__init__()
        self.num_classes = num_classes
        self.layers = self._make_layers(cfgs[vgg_type])
        
    def forward(self,x):
        x = self.layers(x)
        return x
    
    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if type(x)==int:
                out_channels = x
                layers+=[nn.Conv2d(in_channels,out_channels,3,padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels=out_channels
            elif x == 'M':
                layers+= [nn.MaxPool2d((2,2))]
        
        layers+=[
                 nn.Flatten(),
                 nn.Linear(7*7*512,4096),
                 nn.Dropout(0.5),
                 nn.Linear(4096,4096),
                 nn.Dropout(0.5),
                 nn.Linear(4096,self.num_classes)]

        return nn.Sequential(*layers)
            
def vgg11(num_classes):
    return VGG('VGG11',num_classes)

def vgg13(num_classes):
    return VGG('VGG13',num_classes)

def vgg16(num_classes):
    return VGG('VGG16',num_classes)

def vgg19(num_classes):
    return VGG('VGG19',num_classes)

