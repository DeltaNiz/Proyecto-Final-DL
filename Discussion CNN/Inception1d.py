import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

def calculate_padding(kernel_size):
    """
    Calculate the padding size for 'same' padding in PyTorch.
    
    Args:
    - kernel_size (int or tuple): Size of the convolutional kernel
    
    Returns:
    - padding (int or tuple): Padding size to achieve 'same' padding
    """
    if isinstance(kernel_size, int):
        padding = kernel_size // 2
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        raise ValueError("Invalid kernel size format")
    
    return padding

class Inception_Module(nn.Module):
    def __init__(self,in_channels,intermediate_channels, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        self.tower1_conv1 = nn.Conv1d(in_channels, intermediate_channels[0], kernel_size=1,padding='same')
        self.tower1_bn1 = nn.BatchNorm1d(intermediate_channels[0]) if use_batch_norm else nn.Identity()
        self.tower1_conv2 = nn.Conv1d(intermediate_channels[0], intermediate_channels[1], kernel_size=9, padding='same')
        self.tower1_bn2 = nn.BatchNorm1d(intermediate_channels[1]) if use_batch_norm else nn.Identity()

        self.tower2_conv1 = nn.Conv1d(in_channels, intermediate_channels[2], kernel_size=1,padding='same')
        self.tower2_bn1 = nn.BatchNorm1d(intermediate_channels[2]) if use_batch_norm else nn.Identity()
        self.tower2_conv2 = nn.Conv1d(intermediate_channels[2], intermediate_channels[3], kernel_size=25, padding='same')
        self.tower2_bn2 = nn.BatchNorm1d(intermediate_channels[3]) if use_batch_norm else nn.Identity()

        self.tower3_conv1 = nn.Conv1d(in_channels, intermediate_channels[4], kernel_size=1,padding='same')
        self.tower3_bn1 = nn.BatchNorm1d(intermediate_channels[4]) if use_batch_norm else nn.Identity() if use_batch_norm else nn.Identity()

        #self.tower4_pool1=nn.MaxPool1d(3,stride=1,padding=calculate_padding(3))
        self.tower4_conv1 = nn.Conv1d(in_channels, intermediate_channels[5], kernel_size=1,padding='same')
        self.tower4_bn1 = nn.BatchNorm1d(intermediate_channels[5]) if use_batch_norm else nn.Identity()

    def forward(self,x):
        tower1=F.relu(self.tower1_bn1(self.tower1_conv1(x)))
        tower1=F.relu(self.tower1_bn2(self.tower1_conv2(tower1)))
        tower2=F.relu(self.tower2_bn1(self.tower2_conv1(x)))
        tower2=F.relu(self.tower2_bn2(self.tower2_conv2(tower2)))
        tower3=F.relu(self.tower3_bn1(self.tower3_conv1(x)))
        tower4=F.max_pool1d(x,kernel_size=3,stride=1,padding=calculate_padding(3))
        tower4=F.relu(self.tower4_bn1(self.tower4_conv1(tower4)))
        return torch.cat([tower1, tower2, tower3, tower4], dim=1)    
    
class Inception_CNN(nn.Module):
    def __init__(self,out_dim=20):
        super().__init__()
        self.conv1=nn.Conv1d(1,64,47)
        self.bn1=nn.BatchNorm1d(64)
        self.conv2=nn.Conv1d(64,128,29)
        self.bn2=nn.BatchNorm1d(128)
        self.incep1=Inception_Module(128,[96,128,16,16,64,32],use_batch_norm=True)
        self.incep2=Inception_Module(240,[128,156,32,96,64,64],use_batch_norm=True)
        self.incep3=Inception_Module(380,[96,208,16,48,160,64],use_batch_norm=True)
        self.incep4=Inception_Module(480,[112,192,24,64,128,96],use_batch_norm=False)
        self.incep5=Inception_Module(480,[144,272,32,64,112,64],use_batch_norm=False)
        self.incep6=Inception_Module(512,[144,288,32,64,112,64],use_batch_norm=False)
        self.conv3=nn.Conv1d(528,160,1)
        self.flat=nn.Flatten()
        self.fc1=nn.Linear(1600,768)
        self.dropout1=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(768,256)
        self.dropout2=nn.Dropout(p=0.5)
        self.out=nn.Linear(256,out_dim)
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.max_pool1d(x,4)
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.max_pool1d(x,3)
        x=self.incep2(self.incep1(x))
        x=F.max_pool1d(x,4)
        x=self.incep5(self.incep4(self.incep3(x)))
        x=F.max_pool1d(x,4)
        x=F.avg_pool1d(self.incep6(x),3)
        x=F.relu(self.conv3(x))
        x=self.flat(x)
        x=self.dropout1(F.relu(self.fc1(x)))
        x=self.dropout2(F.relu(self.fc2(x)))
        x=self.out(x)
        return x
