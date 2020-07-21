import torch 
from torch import nn 
from torch.nn import functional as F 


class SeperableConv2d_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SeperableConv2d_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch,bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        return x

class Conv2d_Block(nn.Module):
	def __init__(self,in_ch,out_ch,kernel_size,stride,padding,bias,activation=False):
		super(Conv2d_Block,self).__init__()
		self.activation = activation
		self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,bias=bias)
		self.bn1 = nn.BatchNorm2d(out_ch)

	def forward(self,x):
		x = self.conv1(x)
		x = self.bn1(x)
		if self.activation:
			x = F.relu(x)
		
		return x

class Entry_flow(nn.Module):
	def __init__(self):
		super(Entry_flow,self).__init__()
		self.conv1 = Conv2d_Block(3,32,kernel_size=3,stride=2,padding=1,bias=False,activation=True)
		self.conv2 = Conv2d_Block(32,64,kernel_size=3,stride=1,padding=1,bias=False,activation=True)

		self.sep_conv1 = SeperableConv2d_Block(64,128)
		self.actv1 = nn.ReLU(inplace=True)
		self.sep_conv2 = SeperableConv2d_Block(128,128)
		self.max_pool1 = nn.MaxPool2d(3,stride=2,padding=1)
		self.shortcut1 = Conv2d_Block(64,128,kernel_size=1,stride=2,padding=0,bias=False)

		self.actv2 = nn.ReLU(inplace=True)
		self.sep_conv3 = SeperableConv2d_Block(128,256)
		self.actv3 = nn.ReLU(inplace=True)
		self.sep_conv4 = SeperableConv2d_Block(256,256)
		self.max_pool2 = nn.MaxPool2d(3,stride=2,padding=1)
		self.shortcut2 = Conv2d_Block(128,256,kernel_size=1,stride=2,padding=0,bias=False)


		self.actv4 = nn.ReLU(inplace=True)
		self.sep_conv5 = SeperableConv2d_Block(256,728)
		self.actv5 = nn.ReLU(inplace=True)
		self.sep_conv6 = SeperableConv2d_Block(728,728)
		self.max_pool3 = nn.MaxPool2d(3,stride=2,padding=1)
		self.shortcut3 = Conv2d_Block(256,728,kernel_size=1,stride=2,padding=0,bias=False)

	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		
		s = self.shortcut1(x)
		x = self.sep_conv1(x)
		x = self.actv1(x)
		x = self.sep_conv2(x)
		x = self.max_pool1(x)
		x = torch.add(x,s)
		
		s = self.shortcut2(x)
		x = self.actv2(x)
		x = self.sep_conv3(x)
		x = self.actv3(x)
		x = self.sep_conv4(x)
		x = self.max_pool2(x)
		x = torch.add(x,s)

		s = self.shortcut3(x)
		x = self.actv4(x)
		x = self.sep_conv5(x)
		x = self.actv5(x)
		x = self.sep_conv6(x)
		x = self.max_pool3(x)
		x = torch.add(x,s)

		return x 


class Middle_flow(nn.Module):
	def __init__(self):
		super(Middle_flow,self).__init__()
		self.actv1 = nn.ReLU(inplace=True)
		self.sep_conv1 = SeperableConv2d_Block(728,728)
		self.actv2 = nn.ReLU(inplace=True)
		self.sep_conv2 = SeperableConv2d_Block(728,728)
		self.actv3 = nn.ReLU(inplace=True)
		self.sep_conv3 = SeperableConv2d_Block(728,728)

	def forward(self,x):
		s = x 
		x = self.actv1(x)
		x = self.sep_conv1(x)
		x = self.actv2(x)
		x = self.sep_conv2(x)
		x = self.actv3(x)
		x = self.sep_conv3(x)

		x = torch.add(x,s)

		return x


class Exit_flow(nn.Module):
	def __init__(self):
		super(Exit_flow,self).__init__()
		self.actv1 = nn.ReLU(inplace=True)
		self.sep_conv1 = SeperableConv2d_Block(728,728)
		self.actv2 = nn.ReLU(inplace=True)
		self.sep_conv2 = SeperableConv2d_Block(728,1024)
		self.max_pool1 = nn.MaxPool2d(3,stride=2,padding=1)

		self.sep_conv3 = SeperableConv2d_Block(1024,1536)
		self.actv3 = nn.ReLU(inplace=True)

		self.sep_conv4 = SeperableConv2d_Block(1536,2048)
		self.actv4 = nn.ReLU(inplace=True)

		self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

	def forward(self,x):
		x = self.actv1(x)
		x = self.sep_conv1(x)
		x = self.actv2(x)
		x = self.sep_conv2(x)
		x = self.max_pool1(x)

		x = self.sep_conv3(x)
		x = self.actv3(x)

		x = self.sep_conv4(x)
		x = self.actv4(x)

		x = self.avg_pool(x)

		return x


class xception(nn.Module):
	def __init__(self,n_classes):
		super(xception,self).__init__()
		self.n_classes = n_classes
		self.entry_flow = Entry_flow()
		self.middle_flow = nn.Sequential(*[Middle_flow() for _ in range(8)])
		self.exit_flow = Exit_flow()
		self.fc1 = nn.Linear(2048,n_classes)

	def forward(self,x):
		x = self.entry_flow(x)
		x = self.middle_flow(x)
		x = self.exit_flow(x)
		x = x.view(-1,2048)
		x = self.fc1(x)
		return x 
