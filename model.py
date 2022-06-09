import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3(nn.Module):
    def __init__(self,class_n = 80):
        super(YOLOv3 , self).__init__()
        self.class_n = class_n
        self.first_block = nn.Sequential(
                                    nn.Conv2d(3,32,3,1,1),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(32,64,3,2,1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(),
                                    )

        self.res_1 = self.Res(64)
        self.conv_1 = nn.Conv2d(64,128,3,2,1)

        self.res_2 = nn.Sequential(self.Res(128),self.Res(128))
        self.conv_2 = nn.Conv2d(128,256,3,2,1)
       
        self.res_3 = nn.Sequential(self.Res(256),self.Res(256),self.Res(256),self.Res(256),self.Res(256),self.Res(256),self.Res(256),self.Res(256))
        self.conv_3 = nn.Conv2d(256,512,3,2,1)

        self.res_4 = nn.Sequential(self.Res(512),self.Res(512),self.Res(512),self.Res(512),self.Res(512),self.Res(512),self.Res(512),self.Res(512))
        self.conv_4 = nn.Conv2d(512,1024,3,2,1)
   
        self.res_5 = nn.Sequential(self.Res(1024),self.Res(1024),self.Res(1024),self.Res(1024))
        self.conv_block = nn.Sequential(self.Res(1024),self.Res(1024),self.Res(1024))

        self.scale1_convblock = nn.Sequential(nn.Sequential(
                            nn.Conv2d(384,128,1,1), 
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(),
                            nn.Conv2d(128,256,3,1,1),
                            nn.BatchNorm2d(256),
                            nn.LeakyReLU(),
                            ),self.Res(256),self.Res(256))

        self.scale2_convblock = nn.Sequential(nn.Sequential(
                            nn.Conv2d(768,256,1,1), 
                            nn.BatchNorm2d(256),
                            nn.LeakyReLU(),
                            nn.Conv2d(256,512,3,1,1),
                            nn.BatchNorm2d(512),
                            nn.LeakyReLU(),
                            ),self.Res(512),self.Res(512),)

        self.scale1_output = nn.Conv2d(256,(3 * (5 +self.class_n)),1,1)
        self.scale2_output = nn.Conv2d(512,(3 * (5 +self.class_n)),1,1)
        self.scale3_output = nn.Conv2d(1024,(3 * (5+self.class_n)),1,1)

        self.Up = nn.PixelShuffle(upscale_factor=2)
        
    def Res(self,ch):
        block = nn.Sequential(nn.Conv2d(ch,int(ch/2),1,1), nn.BatchNorm2d(int(ch/2)), nn.LeakyReLU(), nn.Conv2d(int(ch/2),ch,3,1,1), nn.BatchNorm2d(ch), nn.LeakyReLU())
        return block
 
    def forward(self,x):
        x = self.first_block(x)
        res = self.res_1(x)
        x = x + res
        x = self.conv_1(x)
        
        for layer in self.res_2:
            res = layer(x)
            x = x + res
        x = self.conv_2(x) 

        for layer in self.res_3:
            res = layer(x)
            x = x + res       
        x3 = x
        x = self.conv_3(x)

        for layer in self.res_4:
            res = layer(x)
            x = x + res
        x4 = x
        x = self.conv_4(x)
   
        for layer in self.res_5:
            res = layer(x)
            x = x + res
   
        for layer in self.conv_block:
            x = layer(x)
   
        scale3_result = self.scale3_output(x)
   
        scale2_up = self.Up(x)
        x = torch.cat([x4 , scale2_up],dim = 1)
        for layer in self.scale2_convblock :
            x = layer(x)
        x4 = x
        scale2_result = self.scale2_output(x)

        scale1_up = self.Up(x4)
        x = torch.cat([x3 , scale1_up],dim = 1)
        for layer in self.scale1_convblock :
            x = layer(x)
        scale1_result = self.scale1_output(x)

        return  scale3_result , scale2_result , scale1_result 