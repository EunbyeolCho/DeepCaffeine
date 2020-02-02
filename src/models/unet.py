#github usuyama/pytorch_unet 참고
#resnet50 based

import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, padding =1),
    nn.ELU(inplace = True),
    nn.Dropout(p=0.1),
    nn.Conv2d(out_channels, out_channels, 3, padding=1),
    nn.ELU(inplace = True),
    nn.Dropout(p=0.1)
    )
    
class UNet(nn.Module):
  
  def __init__(self, n_class):
    super().__init__()
    
    self.dconv_down1 = double_conv(1, 64)
    self.dconv_down2 = double_conv(64, 128)
    self.dconv_down3 = double_conv(128, 256)
    self.dconv_down4 = double_conv(256, 512)        

    self.maxpool = nn.MaxPool2d(2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
    
    self.dconv_up3 = double_conv(256 + 512, 256)
    self.dconv_up2 = double_conv(128 + 256, 128)
    self.dconv_up1 = double_conv(128 + 64, 64)
      
    self.conv_last = nn.Conv2d(64, n_class, 1)
    
    
  def forward(self, x):
      #print("input x : ", x.shape)
      conv1 = self.dconv_down1(x)
      #print("conv1 : ", conv1.shape)
      x = self.maxpool(conv1)
      #print("1st maxpool : ", x.shape)

      conv2 = self.dconv_down2(x)
      #print("conv2 : ", conv2.shape)
      x = self.maxpool(conv2)
      #print("2nd maxpool : ", x.shape)

      conv3 = self.dconv_down3(x)
      #print("conv3 : ", conv3.shape)
      x = self.maxpool(conv3) 
      #print("3rd maxpool : ", x.shape)  
      
      x = self.dconv_down4(x)
      #print("conv4 : ", x.shape)
      
      x = self.upsample(x)     
      #print("unsample : ", x.shape)   
      x = torch.cat([x, conv3], dim=1)
      
      x = self.dconv_up3(x)
      x = self.upsample(x)        
      x = torch.cat([x, conv2], dim=1)       

      x = self.dconv_up2(x)
      x = self.upsample(x)        
      x = torch.cat([x, conv1], dim=1)   
      
      x = self.dconv_up1(x)
      
      out = self.conv_last(x)
      
      return out
