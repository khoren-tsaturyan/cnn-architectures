from torch import nn
from torch.nn import functional as F



class alexnet(nn.Module):
    def __init__(self,num_classes):
        super(alexnet,self).__init__()
        self.conv1 = nn.Conv2d(3,96,kernel_size=11,stride=4)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 = nn.Conv2d(96,256,kernel_size=5,padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1)
            
        self.fc1 = nn.Linear(13*13*256,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.out = nn.Linear(4096,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = x.view(-1,13*13*256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)        
        

     

     

