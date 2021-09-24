import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# i made a r2conv block because it was cleaner and reusable
class R2conv_block(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(R2conv_block, self).__init__()
        
        #Contains the input 1x1 conv added with the output of the second recurrent layer (Recurrect Residual Conv unit)
        
        self.conv1x1 = nn.Conv2d(input_channels, output_channels, 1)
        self.seq = nn.Sequential(nn.Conv2d(output_channels,output_channels,3,padding=1),
                                        nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True))
    def forward(self, x):
        
        Input = self.conv1x1(x) 
        
        #First Recurrent layer
        rec1 = self.seq(Input)
        x = self.seq(Input + rec1)
        
        #Second Recurrent layer
        rec2 = self.seq(x)
        x = self.seq(x + rec2)
        return Input + x #residual connection
        
        
        

class R2UNet(nn.Module):
  
    def __init__(self):
        super(R2UNet, self).__init__()
        
        #define the blocks for your model (in task 1 i redefined pool layers everytime which was a mistake)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.down_R2conv1 = R2conv_block(3, 64)
        
        self.down_R2conv2 = R2conv_block(64, 128)
        
        self.down_R2conv3 = R2conv_block(128, 256)
    
        self.down_R2conv4 = R2conv_block(256, 512)
        
        self.down_R2conv5 = R2conv_block(512, 1024)
        
        # upconvolution layer which includes the upsample, conv, batchnorm(for fast convergence) and relu(define relu here so 
        # we dont have to put relu in the forward function)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear'), 
                                 nn.Conv2d(1024, 512 ,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace = True))
        self.up_R2conv4 = R2conv_block(1024, 512)
        
        self.up3 = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear'), 
                                 nn.Conv2d(512, 256, 3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace = True))
        self.up_R2conv3 = R2conv_block(512, 256)
            
        self.up2 = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear'), 
                                 nn.Conv2d(256, 128, 3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace = True))
        self.up_R2conv2 = R2conv_block(256, 128)
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear'), 
                                 nn.Conv2d(128, 64, 3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace = True))
        self.up_R2conv1 = R2conv_block(128, 64)
        
        self.conv1x1 = nn.Conv2d(64, 34, 1)
        
    def forward(self, x):
        
        #going down (encoding)
        x1 = self.down_R2conv1(x)
        x = self.pool(x1)
        
        x2 = self.down_R2conv2(x)
        x = self.pool(x2)
        
        x3 = self.down_R2conv3(x)
        x = self.pool(x3)
        
        x4 = self.down_R2conv4(x)
        x = self.pool(x4)
        
        x5 = self.down_R2conv5(x)
        
        #going up (decoding)
        
        x = self.up4(x5)
        x = torch.cat((x4, x), dim = 1)
        x =  self.up_R2conv4(x)
        
        x = self.up3(x)
        x = torch.cat((x3, x), dim = 1)
        x = self.up_R2conv3(x)
        
        x = self.up2(x)
        x = torch.cat((x2, x), dim = 1)
        x =  self.up_R2conv2(x)
        
        x = self.up1(x)
        x = torch.cat((x1, x), dim = 1)
        x =  self.up_R2conv1(x)
        
        x = self.conv1x1(x)
        
        return x